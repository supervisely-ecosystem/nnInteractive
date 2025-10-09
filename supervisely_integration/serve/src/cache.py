import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from cachetools import TTLCache
from fastapi import UploadFile

from supervisely._utils import rand_str
from supervisely.api.api import Api
from supervisely.api.volume.volume_api import VolumeInfo
from supervisely.io.fs import silent_remove
import supervisely.io.env as sly_env
from supervisely.nn.inference.cache import InferenceImageCache, PersistentImageTTLCache
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger


class PersistentVolumeTTLCache(PersistentImageTTLCache):

    def save_volume(self, key: Any, source: Union[str, Path]) -> None:
        if not self._base_dir.exists():
            self._base_dir.mkdir()

        filepath = self._base_dir / Path(f"volume_{key}.nrrd")
        self[key] = filepath

        if filepath.exists():
            logger.debug(f"Rewrite volume {str(filepath)}")
        if isinstance(source, bytes):
            with open(filepath, "wb") as f:
                f.write(source)
        else:
            with open(source, "rb") as fsrc, open(filepath, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

    def get_volume_path(self, key: Any) -> Path:
        return self[key]


class InferenceVolumeCache(InferenceImageCache):
    class _LoadType(Enum):
        ImageId: str = "IMAGE"
        ImageHash: str = "HASH"
        Frame: str = "FRAME"
        Video: str = "VIDEO"
        Volume: str = "VOLUME"

    def __init__(
        self,
        maxsize: int,
        ttl: int,
        is_persistent: bool = True,
        base_folder: str = sly_env.smart_cache_container_dir(),
        log_progress: bool = False,
    ):
        super().__init__(maxsize, ttl, is_persistent, base_folder, log_progress)
        if is_persistent:
            self._data_dir = Path(base_folder)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._cache = PersistentVolumeTTLCache(maxsize, ttl, self._data_dir)
        else:
            self._cache = TTLCache(maxsize, ttl)

    def cache_task(self, api: Api, state: dict):
        if "server_address" in state and "api_token" in state:
            api = Api(state["server_address"], state["api_token"])
        api.logger.debug("Request state in cache endpoint", extra=state)
        image_ids, task_type = self._parse_state(state)
        kwargs = {"return_images": False}
        if task_type is InferenceVolumeCache._LoadType.Volume:
            volume_id = image_ids
            self.download_volume(api, volume_id, **kwargs)

    def cache_files_task(self, files: List[UploadFile], state: dict):
        logger.debug("Request state in cache endpoint", extra=state)
        image_ids, task_type = self._parse_state(state)

        if task_type is InferenceVolumeCache._LoadType.Volume:
            volume_id = image_ids
            self._wait_if_in_queue(volume_id, logger)
            self._load_queue.set(volume_id, volume_id)
            self.add_volume_to_cache(volume_id, files[0].file)

    def run_cache_task_manually(
        self,
        api: Api,
        list_of_ids_ranges_or_hashes: List[Union[str, int, List[int]]],
        *,
        dataset_id: Optional[int] = None,
        video_id: Optional[int] = None,
        volume_id: Optional[int] = None,
    ) -> None:
        state = {}
        if volume_id is not None:
            api.logger.debug("Got a task to add volume to cache")
            if not isinstance(self._cache, PersistentVolumeTTLCache):
                raise ValueError("Volume can be added only to persistent cache")
            state["volume_id"] = volume_id
        else:
            raise ValueError("Only volume_id is supported in volume cache")
        self._download_executor.submit(self.cache_task, api=api, state=state)

    def set_project_meta(self, project_id, project_meta):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentVolumeTTLCache):
            self._cache.save_project_meta(pr_meta_name, project_meta)
        else:
            self._cache[pr_meta_name] = project_meta

    def get_project_meta(self, api: Api, project_id: int):
        pr_meta_name = self._project_meta_name(project_id)
        if isinstance(self._cache, PersistentVolumeTTLCache):
            if pr_meta_name in self._cache:
                return self._cache.get_project_meta(pr_meta_name)
            project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
            self._cache.save_project_meta(pr_meta_name, project_meta)
            return project_meta
        else:
            if pr_meta_name in self._cache:
                return self._cache[pr_meta_name]
            project_meta = ProjectMeta.from_json(api.project.get_meta(project_id))
            self._cache[pr_meta_name] = project_meta
            return project_meta

    def _parse_state(self, state: dict) -> Tuple[List[Any], _LoadType]:
        if "volume_id" in state:
            return state["volume_id"], InferenceVolumeCache._LoadType.Volume
        raise ValueError("State has no proper fields: 'volume_id'")

    def _volume_name(self, volume_id: int) -> str:
        return f"volume_{volume_id}"

    def download_volume(self, api: Api, volume_id: int, **kwargs):
        name = self._volume_name(volume_id)
        progress_cb = kwargs.get("progress_cb", None)
        volume_info = kwargs.get("volume_info", None)
        if volume_info is None:
            volume_info = api.volume.get_info_by_id(volume_id)
        volume_info: VolumeInfo

        self._wait_if_in_queue(name, api.logger)
        if name not in self._cache:
            download_time = time.monotonic()
            self._load_queue.set(name, volume_id)
            try:
                logger.debug("Downloading volume #%s", volume_id)
                if progress_cb is None and self.log_progress:
                    size = volume_info.sizeb
                    if size is None:
                        size = "unknown"
                    else:
                        size = int(size)

                    prog_n = 0
                    prog_t = time.monotonic()

                    def _progress_cb(n):
                        nonlocal prog_n
                        nonlocal prog_t
                        prog_n += n
                        cur_t = time.monotonic()
                        if cur_t - prog_t > 3 or (isinstance(size, int) and prog_n >= size):
                            prog_t = cur_t
                            percent_str = ""
                            if isinstance(size, int):
                                percent_str = f" ({(prog_n*100) // size}%)"
                            prog_str = (
                                f"{(prog_n / 1000000):.2f}/{(size / 1000000):.2f} MB{percent_str}"
                            )
                            logger.debug(
                                "Downloading volume #%s: %s",
                                volume_id,
                                prog_str,
                            )

                    progress_cb = _progress_cb
                temp_volume_path = Path("/tmp/smart_cache").joinpath(
                    f"_{rand_str(6)}_" + volume_info.name
                )
                api.volume.download_path(volume_id, temp_volume_path, progress_cb=progress_cb)
                api.logger.debug(f"Add volume #{volume_id} to cache")
                self.add_volume_to_cache(volume_id, temp_volume_path)
                api.logger.debug(
                    f"Volume #{volume_id} downloaded to cache in {download_time:.2f} sec",
                    extra={"volume_id": volume_id, "download_time": download_time},
                )
                silent_remove(temp_volume_path)
            except Exception as e:
                self._load_queue.delete(name)
                raise e

        return self._cache.get_volume_path(name)

    def add_volume_to_cache(self, volume_id: int, source: Union[str, Path]) -> None:
        """
        Adds volume to cache.
        """
        if isinstance(self._cache, PersistentVolumeTTLCache):
            with self._lock:
                key = self._volume_name(volume_id)
                self._cache.save_volume(key, source)
                self._load_queue.delete(key)
            logger.debug(f"Volume #{volume_id} added to cache", extra={"volume_id": volume_id})
        else:
            raise ValueError("Volume can be added only to persistent cache")

    def get_volume_path(self, key: Union[int, str]) -> str:
        if isinstance(key, int):
            key = self._volume_name(key)
        return str(self._cache.get_volume_path(key))
