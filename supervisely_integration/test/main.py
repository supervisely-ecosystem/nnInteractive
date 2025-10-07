import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import supervisely as sly
import torch
from dotenv import load_dotenv
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# --- Download Trained Model Weights (~400MB) ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "./test_data"  # Specify the download directory
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

load_dotenv("supervisely.env")
api = sly.Api.from_env()

project_id = 4068
volume_id = 1639941
volume_info = api.volume.get_info_by_id(volume_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
test_cls = project_meta.get_obj_class("test")
if test_cls is None:
    test_cls = sly.ObjClass("test", sly.Mask3D)
    project_meta = project_meta.add_obj_class(test_cls)
    project_meta = api.project.update_meta(project_id, project_meta)


nrrd_path = Path(DOWNLOAD_DIR) / volume_info.name
if not nrrd_path.exists():
    api.volume.download_path(volume_id, str(nrrd_path))
volume_np, volume_meta = sly.volume.read_nrrd_serie_volume_np(str(nrrd_path))


# -----------------------------------------------------------------------------
# --- Download the model from Hugging Face ------------------------------------
# -----------------------------------------------------------------------------
if not Path(DOWNLOAD_DIR, MODEL_NAME).exists():
    from huggingface_hub import snapshot_download
    download_path = snapshot_download(
        repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
    )


# -----------------------------------------------------------------------------
# --- Initialize Inference Session --------------------------------------------
# -----------------------------------------------------------------------------
session = nnInteractiveInferenceSession(
    device=torch.device("cuda:0"),  # Set inference device
    use_torch_compile=False,  # Experimental: Not tested yet
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use all available CPU cores
    do_autozoom=True,  # Enables AutoZoom for better patching
    use_pinned_memory=True,  # Optimizes GPU memory transfers
)

# Load the trained model
model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
session.initialize_from_trained_model_folder(model_path)

# -----------------------------------------------------------------------------
# --- Load Input Image (Example with SimpleITK) -------------------------------
# -----------------------------------------------------------------------------
# DO NOT preprocess the image in any way. Give it to nnInteractive as it is! DO NOT apply level window, DO NOT normalize
# intensities and never ever convert an image with higher precision (float32, uint16, etc) to uint8!
# The ONLY instance where some preprocesing makes sense is if your original image is too large to be reasonably used.
# This may be the case, for example, for some microCT images. In this case you can consider downsampling.
input_image = sitk.ReadImage(nrrd_path)
img = sitk.GetArrayFromImage(input_image)  # (139, 512, 512)
img = img.transpose(2, 1, 0)[None]  # to (1, 512, 512, 139)

# Validate input dimensions
if img.ndim != 4:
    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

session.set_image(img)

# --- Define Output Buffer ---
target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
session.set_target_buffer(target_tensor)

# --- Interacting with the Model ---
# Interactions can be freely chained and mixed in any order. Each interaction refines the segmentation.
# The model updates the segmentation mask in the target buffer after every interaction.

# Example: Add a **positive** point interaction
# POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
# POSITIVE_POINTS = [
#     (284, 275, 80),
#     (206, 320, 80),
#     (248, 325, 80),
#     (217, 261, 80),
#     (234, 242, 80),
#     (266, 263, 80),
# ]  # Example coordinates
# for POINT in POSITIVE_POINTS:
#     session.add_point_interaction(POINT, include_interaction=True)

# NEGATIVE_POINTS = [
#     (285, 371, 80),
#     (327, 270, 80),
#     (271, 210, 80),
#     (167, 200, 80),
#     (150, 335, 80),
#     (207, 377, 80),
# ]  # Example coordinates
# for POINT in NEGATIVE_POINTS:
#     session.add_point_interaction(POINT, include_interaction=False)
# # Example: Add a **negative** point interaction
# # To make any interaction negative set include_interaction=False
# session.add_point_interaction(POINT_COORDINATES, include_interaction=False)

# # Example: Add a bounding box interaction
# # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
# # Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
# # This means that **one dimension must be [d, d+1]** to indicate a single slice.

# # Example of a 2D bounding box in the axial plane (XY slice at depth Z)
# # BBOX_COORDINATES = [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10

# session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)

# # Example: Add a scribble interaction
# # - A 3D image of the same shape as img where one slice (any axis-aligned orientation) contains a hand-drawn scribble.
# # - Background must be 0, and scribble must be 1.
# # - Use session.preferred_scribble_thickness for optimal results.
# session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# # Example: Add a lasso interaction
# # - Similarly to scribble a 3D image with a single slice containing a **closed contour** representing the selection.
LASSO_IMAGE = np.zeros(img.shape[1:], dtype=np.uint8)

ann = api.volume.annotation.download(volume_id)
key_id_map = sly.KeyIdMap()
ann = sly.VolumeAnnotation.from_json(ann, project_meta, key_id_map)
figure = ann.figures[0]
vol_dimensions = volume_info.meta["dimensionsIJK"]  # {'x': 512, 'y': 512, 'z': 139}
geometry: sly.Bitmap = figure.geometry
if figure.plane_name == sly.Plane.AXIAL:
    mask_shape = (vol_dimensions["x"], vol_dimensions["y"])
    mask = geometry.get_mask(mask_shape).astype(np.uint8)
    LASSO_IMAGE[:, :, figure.slice_index] = mask.T
elif figure.plane_name == sly.Plane.CORONAL:
    mask_shape = (vol_dimensions["x"], vol_dimensions["z"])
    mask = geometry.get_mask(mask_shape).astype(np.uint8)
    LASSO_IMAGE[:, figure.slice_index, :] = mask.T
elif figure.plane_name == sly.Plane.SAGITTAL:
    mask_shape = (vol_dimensions["y"], vol_dimensions["z"])
    mask = geometry.get_mask(mask_shape).astype(np.uint8)
    LASSO_IMAGE[figure.slice_index, :, :] = mask.T
else:
    raise ValueError(f"Unsupported plane_name: {figure.plane_name}")
# mask = geometry.get_mask(mask_shape)
# mask_img = img[0, :, :, figure.slice_index]
session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# You can combine any number of interactions as needed.
# The model refines the segmentation result incrementally with each new interaction.

# --- Retrieve Results ---
# The target buffer holds the segmentation result.
results = session.target_buffer.clone()
# OR (equivalent)
results = target_tensor.clone()

# Cloning is required because the buffer will be **reused** for the next object.
# Alternatively, set a new target buffer for each object:
# session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))

# --- Start a New Object Segmentation ---
# session.reset_interactions()  # Clears the target buffer and resets interactions


# -----------------------------------------------------------------------------
# --- Save or Visualize Results ------------------------------------------------
# -----------------------------------------------------------------------------
# Save results as NRRD
# Example: Save results as Sly Annotation
res = results.numpy().astype("uint8")
volume_info = api.volume.get_info_by_id(volume_id)
mask = sly.Mask3D(data=res > 0, volume_header=volume_meta)

# ann = api.volume.annotation.download(volume_id)
obj = sly.VolumeObject(obj_class=test_cls, mask_3d=mask)
volume_ann = sly.VolumeAnnotation(
    volume_info.meta,
    objects=[obj],
    spatial_figures=[obj.figure],
)
api.volume.annotation.append(volume_info.id, volume_ann)
sly.logger.info(
    f"Annotation has been sucessfully uploaded to the volume {volume_info.name} in dataset with ID={volume_info.dataset_id}"
)
