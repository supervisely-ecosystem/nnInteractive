import functools
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk
import torch
from dotenv import load_dotenv
from fastapi import Request, Response, status
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from src.cache import InferenceVolumeCache

import supervisely as sly
import supervisely.io.env as sly_env
import supervisely.io.fs as sly_fs
from supervisely.app.content import get_data_dir
from supervisely.imaging.color import generate_rgb
from supervisely.nn.inference.inference import Inference, InferenceImageCache
from supervisely.nn.utils import CheckpointInfo, ModelSource
from supervisely.sly_logger import logger
from supervisely.volume_annotation.plane import Plane


def send_volume_is_downloading_notification(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        request: Request = args[0]
        context = request.state.context
        api: sly.Api = request.state.api
        volume_id = context["volumeId"]
        try:
            api.post(
                "volumes.notify-annotation-tool",
                data={
                    "type": "volumes:volume-downloading-started",
                    "data": {
                        "volumeId": volume_id,
                    },
                },
            )
            value = func(*args, **kwargs)
        except Exception as exc:
            raise exc
        finally:
            api.post(
                "volumes.notify-annotation-tool",
                data={
                    "type": "volumes:volume-downloading-completed",
                    "data": {
                        "volumeId": volume_id,
                    },
                },
            )
        return value

    return wrapper


@send_volume_is_downloading_notification
def download_volume_from_context(
    request: Request,
    context: dict,
    api: sly.Api,
    cache: InferenceImageCache = None,
) -> str:
    if "volume_id" in context:
        if cache is not None:
            cache.download_volume(api, context["volume_id"])

        volume_path = cache.get_volume_path(context["volume_id"])
        return volume_path
    elif "volume" in context:
        volume_id = context["volume"]["volume_id"]
        if cache is not None:
            cache.download_volume(api, volume_id)
        volume_path = cache.get_volume_path(volume_id)
        return volume_path
    else:
        raise Exception("Project type is not supported")


def get_hash_from_context(context: dict):
    if "volume" in context:
        volume_id = context["volume"]["volume_id"]
        # slice_index = context["volume"]["slice_index"]
        # normal = context["volume"]["normal"]
        # window_center = context["volume"]["window_center"]
        # window_width = context["volume"]["window_width"]
        # plane = sly.Plane.get_name(normal)
        # return "_".join(map(str, [volume_id, slice_index, plane, window_center, window_width]))
        return str(volume_id)
    else:
        raise Exception("Project type is not supported")


class PredictionVolumeMask3D(sly.nn.Prediction):
    def __init__(self, class_name: str, volume_mask_3d: sly.Mask3D):
        super(PredictionVolumeMask3D, self).__init__(class_name=class_name)
        self.volume_mask_3d = volume_mask_3d


class nnInteractiveSlyInference(Inference):
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
    FRAMEWORK_NAME = "nnInteractive"

    def __init__(self, *args, **kwargs):
        self.class_names = ["object_mask"]
        self.mask_colors = [[0, 255, 0]]
        self._model_meta = None
        self.checkpoint_info = None
        self.process_volume = False
        self._init_mask_cache = {}  # cache for init masks for images
        super(nnInteractiveSlyInference, self).__init__(*args, **kwargs)
        self.cache = InferenceVolumeCache(
            maxsize=sly_env.smart_cache_size(),
            ttl=sly_env.smart_cache_ttl(),
            is_persistent=True,
            base_folder=sly_env.smart_cache_container_dir(),
            log_progress=True,
        )
        self._inference_image_lock = self.cache._lock

    def _download_pretrained_model(
        self, model_files: dict, log_progress: bool = True, headless: bool = True
    ):
        local_model_files = {}
        cache_dir = self._checkpoints_cache_dir()

        file_path = model_files["checkpoint"]
        repo_id, file_name = file_path.rsplit("/", 1)
        model_path = Path(self.model_dir) / file_name
        cached_path = Path(cache_dir) / file_name

        if model_path.exists():
            local_model_files[file_name] = str(model_path)
            logger.debug(f"Model: '{file_name}' was found in model dir")
            return local_model_files
        if cached_path.exists():
            local_model_files[file_name] = str(cached_path)
            logger.debug(f"Model: '{file_name}' was found in checkpoint cache")
            return local_model_files

        from huggingface_hub import snapshot_download

        logger.debug(f"Model: '{file_name}' was found in model dir")
        snapshot_download(
            repo_id=repo_id, allow_patterns=[f"{file_name}"], local_dir=self.model_dir
        )
        local_model_files[file_name] = str(model_path)

        if log_progress:
            if self.gui is not None:
                self.gui.download_progress.hide()
        return local_model_files

    def load_on_device():
        pass

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        if model_source == ModelSource.CUSTOM:
            # self.class_names = ["object_mask"]  # TODO: get class names from custom model
            checkpoint_path = self._prepare_custom_model(model_files)
        else:
            self.class_names = ["object_mask"]
            checkpoint_path = self._prepare_pretrained_model(model_files, model_info)

        self.model = nnInteractiveInferenceSession(
            device=torch.device(device),
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use all available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        self.model.initialize_from_trained_model_folder(checkpoint_path)

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta([sly.ObjClass(self.class_names[0], sly.Mask3D)])
        return self._model_meta

    # Utils -------------------- #
    def _prepare_custom_model(self, model_files: dict):
        checkpoint_path = model_files["checkpoint"]
        return checkpoint_path

    def _prepare_pretrained_model(self, model_files: dict, model_info: dict):
        checkpoint_path = model_files["checkpoint"]
        model_name = model_info["meta"]["model_name"]

        obj_classes = [sly.ObjClass(name, sly.Mask3D) for name in self.class_names]
        self._model_meta = sly.ProjectMeta(obj_classes=obj_classes)
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_name,
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=ModelSource.PRETRAINED,
        )
        return checkpoint_path

    def get_classes(self) -> List[str]:
        return self.class_names

    def add_content_to_pretrained_tab(self, gui):
        # TODO: add pretrained model info
        # self.use_bbox = Switch(switched=True)
        # use_bbox_field = Field(
        #     content=self.use_bbox,
        #     title="Use bounding box prompt",
        #     description=(
        #         "Define whether to use bounding box prompt when labeling images and videos or not. "
        #         "If turned off, then only point prompts (positive and negative clicks) will be used. "
        #         "Adding bounding box prompt can be useful when labeling entire objects, while using "
        #         "only point prompts can be better when segmenting specific parts of objects."
        #     ),
        # )
        # return use_bbox_field
        pass

    def support_custom_models(self):
        return True

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        # prepare input data
        input_image = sly.image.read(image_path)
        slice_index = settings.get("slice_index", None)
        # list for storing preprocessed masks
        predictions = []
        if self._model_meta is None:
            self._model_meta = self.model_meta
        if settings["mode"] == "bbox":
            if "rectangle" not in settings:
                bbox_coordinates = settings["bbox_coordinates"]
            else:
                rectangle = sly.Rectangle.from_json(settings["rectangle"])
                bbox_coordinates = [
                    rectangle.top,
                    rectangle.left,
                    rectangle.bottom,
                    rectangle.right,
                ]
            # transform bbox from yxyx to format [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10
            bbox_coordinates = [
                [bbox_coordinates[1], bbox_coordinates[3]],
                [bbox_coordinates[0], bbox_coordinates[2]],
                [slice_index, slice_index + 1],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"]
            object_class = self._model_meta.get_obj_class(class_name)
            if object_class is None:
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Mask3D)
                self._model_meta = self._model_meta.add_obj_class(new_class)
            elif object_class.geometry_type != sly.Mask3D:
                class_name = class_name + "_mask3d"
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Mask3D)
                self._model_meta = self._model_meta.add_obj_class(new_class)

            # get predicted mask
            self.model.reset_interactions()
            # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.add_bbox_interaction(bbox_coordinates, include_interaction=True)
            predictions.append(
                PredictionVolumeMask3D(
                    class_name=class_name,
                    volume_mask_3d=self.model.target_buffer.numpy().astype("uint8"),
                )
            )
        elif settings["mode"] == "points":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # set class name
            if settings.get("points_class_name") not in [None, "None"]:
                class_name = settings["points_class_name"]
            else:
                class_name = self.class_names[0]
            # add new class to model meta if necessary
            if not self._model_meta.get_obj_class(class_name):
                color = generate_rgb(self.mask_colors)
                self.mask_colors.append(color)
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Mask3D, color)
                self._model_meta = self._model_meta.add_obj_class(new_class)

            # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for POINT, LABEL in zip(point_coordinates, point_labels):
                self.model.add_point_interaction(POINT, include_interaction=bool(LABEL))
            results = self.model.target_buffer.clone()
            results = results.numpy().astype("uint8")
            predictions.append(
                PredictionVolumeMask3D(class_name=class_name, volume_mask_3d=results)
            )
        elif settings["mode"] == "combined":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # get bbox coordinates
            bbox_coordinates = settings["bbox_coordinates"]
            # transform bbox from yxyx to xxyyzz format [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10
            bbox_coordinates = [
                [bbox_coordinates[1], bbox_coordinates[3]],
                [bbox_coordinates[0], bbox_coordinates[2]],
                [slice_index, slice_index + 1],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"] + "_mask"
            object_class = self._model_meta.get_obj_class(class_name)
            if object_class is None:
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            elif object_class.geometry_type != sly.Mask3D:
                class_name = class_name + "_mask3d"
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Mask3D)
                self._model_meta = self._model_meta.add_obj_class(new_class)

            # get predicted masks
            self.model.reset_interactions()

            # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.add_bbox_interaction(bbox_coordinates, include_interaction=True)
            for POINT, LABEL in zip(point_coordinates, point_labels):
                self.model.add_point_interaction(POINT, include_interaction=bool(LABEL))
            results = self.model.target_buffer.clone()
            results = results.numpy().astype("uint8")
            predictions.append(
                PredictionVolumeMask3D(class_name=class_name, volume_mask_3d=results)
            )
        return predictions

    def get_smarttool_input(self, figure: sly.FigureInfo):
        if figure.meta is None:
            return None
        smarttool_input = figure.meta.get("smartToolInput", None)
        if smarttool_input is None:
            return None
        crop = smarttool_input.get("crop")
        if crop:
            crop = [*crop[0], *crop[1]]
        positive = smarttool_input["positive"]
        negative = smarttool_input["negative"]
        visible = smarttool_input["visible"]
        return crop, positive, negative, visible

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        @send_error_data
        def smart_segmentation(response: Response, request: Request):
            # 1. parse request
            # 2. download image
            # 3. make crop
            # 4. predict

            logger.debug(
                f"smart_segmentation inference: context=",
                extra={**request.state.context},
            )

            try:
                state = request.state.state
                settings = self._get_inference_settings(state)
                smtool_state = request.state.context
                self.process_volume = smtool_state.get("volume") is not None
                api = request.state.api
                positive_clicks, negative_clicks = (
                    smtool_state["positive"],
                    smtool_state["negative"],
                )
                if len(positive_clicks) + len(negative_clicks) == 0:
                    logger.warning("No clicks received.")
                    response = {
                        "origin": None,
                        "bitmap": None,
                        "success": True,
                        "error": None,
                    }
                    return response
            except Exception as exc:
                logger.warning("Error parsing request:" + str(exc), exc_info=True)
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"message": "400: Bad request.", "success": False}

            # collect clicks
            uncropped_clicks = [{**click, "is_positive": True} for click in positive_clicks]
            uncropped_clicks += [{**click, "is_positive": False} for click in negative_clicks]

            # download image if needed (using cache)
            app_dir = get_data_dir()
            hash_str = get_hash_from_context(smtool_state)

            if hash_str not in self.cache:
                logger.debug(f"downloading image: {hash_str}")
                volume_path = download_volume_from_context(
                    request,
                    smtool_state,
                    api,
                    self.cache,
                )
                self.cache._cache.save_volume(hash_str, volume_path)
            else:
                logger.debug(f"volume found in cache: {hash_str}")
                volume_path = self.cache.get_volume_path(hash_str)

            self._inference_image_lock.acquire()
            try:
                # predict
                logger.debug("Preparing settings for inference request...")
                settings["mode"] = "points"
                # if self.use_bbox.is_switched() and crop:
                #     settings["mode"] = "combined"
                # else:
                #     settings["mode"] = "points"
                volume_id = smtool_state.get("volume").get("volume_id")
                if self.process_volume:
                    volume_plane = (
                        sly.Plane.get_name(smtool_state.get("volume").get("normal")) or "Unknown"
                    )
                    slice_idx = smtool_state.get("volume").get("slice_index")
                    settings["input_image_id"] = f"{volume_id}_{volume_plane}_{slice_idx}"

                point_coordinates, point_labels = [], []
                for click in uncropped_clicks:
                    point_coordinates.append([click["x"], click["y"]])
                    if click["is_positive"]:
                        point_labels.append(True)
                    else:
                        point_labels.append(False)
                settings["point_coordinates"], settings["point_labels"] = (
                    point_coordinates,
                    point_labels,
                )
                pred_mask = self.predict(volume_path, settings)[0].volume_mask_3d
            finally:
                logger.debug("Predict done")
                self._inference_image_lock.release()
                # sly_fs.silent_remove(image_path)

            if pred_mask.any():
                volume_info = api.volume.get_info_by_id(volume_id)
                mask = sly.Mask3D(data=pred_mask > 0)

                # ann = api.volume.annotation.download(volume_id)
                obj_cls = self.model_meta.get_obj_class(self.get_classes()[0])
                obj = sly.VolumeObject(obj_class=obj_cls, mask_3d=mask)
                volume_ann = sly.VolumeAnnotation(
                    volume_info.meta,
                    objects=[obj],
                    spatial_figures=[obj.figure],
                )
                api.volume.annotation.append(volume_info.id, volume_ann)
                sly.logger.debug(f"Smart segmentation annotation appended to volume {volume_id}")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            else:
                logger.debug(f"Predicted mask is empty.")
                response = {
                    "origin": None,
                    "bitmap": None,
                    "success": True,
                    "error": None,
                }
            return response

        @server.post("/is_online")
        def is_online(response: Response, request: Request):
            response = {"is_online": True}
            return response

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    print("An error occured:")
                    print(traceback.format_exc())
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    volume_id = context["volumeId"]

                    api.post(
                        "volumes.notify-annotation-tool",
                        data={
                            "type": "volumes:smart-segmentation-error",
                            "data": {
                                "volumeId": volume_id,
                                "error": {"message": repr(exc)},
                            },
                        },
                    )
                return value

            return wrapper


load_dotenv("supervisely.env")
load_dotenv("debug.env")
api = sly.Api()
root_source_path = str(Path(__file__).parents[1])
debug_session = bool(os.environ.get("DEBUG_SESSION", False))
model_data_path = os.path.join(root_source_path, "models", "models.json")
UPLOAD_SLEEP_TIME = 0.1
NOTIFY_SLEEP_TIME = 0.1

m = nnInteractiveSlyInference(
    use_gui=False,
    model_dir="app_data",
)


# def clean_data():
#     # delete app data since it is no longer needed
#     sly.fs.remove_dir("prompts")
#     sly.fs.remove_dir("frames")
#     sly.logger.info("Successfully cleaned unnecessary app data")


m.serve()
m.gui._models_table.select_row(1)
# m.app.call_before_shutdown(clean_data)
