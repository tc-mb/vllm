# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs
from vllm.logger import init_logger

from .base import MediaIO
from .image import ImageMediaIO

logger = init_logger(__name__)


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty(
        (num_frames, new_height, new_width, channels), dtype=frames.dtype
    )
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoLoader:
    @classmethod
    @abstractmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError


class VideoLoaderRegistry:
    def __init__(self) -> None:
        self.name2class: dict[str, type] = {}

    def register(self, name: str):
        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    @staticmethod
    def load(cls_name: str) -> VideoLoader:
        cls = VIDEO_LOADER_REGISTRY.name2class.get(cls_name)
        assert cls is not None, f"VideoLoader class {cls_name} not found"
        return cls()


VIDEO_LOADER_REGISTRY = VideoLoaderRegistry()


@VIDEO_LOADER_REGISTRY.register("opencv")
class OpenCVVideoBackend(VideoLoader):
    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames and fps
        # - the minimum of the two will be used
        num_frames_to_sample = total_frames_num
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, total_frames_num)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

        if num_frames_to_sample == total_frames_num:
            frame_idx = list(range(0, num_frames_to_sample))
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frames_num - 1, num_frames_to_sample, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_idx), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(max(frame_idx) + 1):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_idx:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        assert i == num_frames_to_sample, (
            f"Expected reading {num_frames_to_sample} frames, "
            f"but only loaded {i} frames from video."
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        # NOTE(Isotr0py): For models like Qwen3-VL/GLM4.5V, this metadata
        # can cause incorrect timestamp calculation without num_frames=-1.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv",
            "frames_indices": list(frame_idx),
            # extra field used to control hf processor's video
            # sampling behavior
            "do_sample_frames": num_frames_to_sample == total_frames_num,
        }

        return frames, metadata


@VIDEO_LOADER_REGISTRY.register("opencv_dynamic")
class OpenCVDynamicVideoBackend(OpenCVVideoBackend):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = 2,
        max_duration: int = 300,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # resample video to target num_frames
        max_frame_idx = total_frames_num - 1
        duration = duration or round(max_frame_idx / original_fps) + 1

        # Refer to:
        # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/glm4v/video_processing_glm4v.py#L103-L140
        frame_indices: range | list[int]
        if duration <= max_duration:
            n = int(math.floor(duration * fps))
            frame_indices = sorted(
                {
                    min(max_frame_idx, int(math.ceil(i * original_fps / fps)))
                    for i in range(n)
                }
            )
        else:
            num_samples = int(max_duration * fps)
            if num_samples >= total_frames_num:
                frame_indices = range(total_frames_num)
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices = sorted(
                    {
                        min(max_frame_idx, int(math.ceil(t * original_fps)))
                        for t in target_seconds
                    }
                )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(total_frames_num):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        assert i == len(frame_indices), (
            f"Expected reading {len(frame_indices)} frames, "
            f"but only loaded {i} frames from video."
        )

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": list(frame_indices),
            "do_sample_frames": False,
        }

        return frames, metadata

@VIDEO_LOADER_REGISTRY.register("enhanced_opencv")
class High_Refresh_VideoLoader(OpenCVVideoBackend):
    
    DOUBLE_FRAME_DURATION : int = 30
    MAX_NUM_FRAMES : int = 30
    MAX_NUM_PACKING : int = 3
    TIME_SCALE : int = 0.1 

    def map_to_nearest_scale(
            self,
            values: np.ndarray,
            scale: np.ndarray) -> np.ndarray:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        # Mapping values to the nearest scale 
        # (efficient large-scale version)
        if cKDTree is None:
            # Fallback method if scipy is not available
            scale_array = np.asarray(scale)
            values_array = np.asarray(values)
            indices = []
            for value in values_array:
                distances = np.abs(scale_array - value)
                indices.append(np.argmin(distances))
            return scale_array[indices]
        
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

    def group_array(self,
                    arr: np.ndarray,
                    size: int):
        return [arr[i:i+size] 
                for i in range(0, len(arr), size)]


    def uniform_sample(
                    self,
                    l : int,
                    n : int) -> list[int]:
        gap = l / n
        return [int(i * gap + gap / 2) for i in range(n)]


    @classmethod
    def load_bytes(cls,
                   data: bytes,
                   num_frames: int = -1,
                   **kwargs,
                   ) -> tuple[npt.NDArray, dict[str, Any]]:
        choose_fps = kwargs.get('choose_fps', None)
        import cv2
        from io import BytesIO
        packing_nums : int = 1

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        # Compatible with Common Sampling Parameters
        if choose_fps and original_fps > 0:
            if duration < cls().DOUBLE_FRAME_DURATION and choose_fps <=5:
                choose_fps = choose_fps * 2
                packing_nums = 2
                choose_frames_nums = round(
                    min(choose_fps, round(original_fps)) *
                    min(cls().MAX_NUM_FRAMES, duration)
                )
                choose_frames_nums = min(
                    choose_frames_nums,
                    round(cls().MAX_NUM_FRAMES * cls().MAX_NUM_PACKING)
                )
            elif choose_fps * int(duration) <= cls().MAX_NUM_FRAMES: # 40
                packing_nums = 1
                choose_frames_nums = round(
                    min(choose_fps, round(original_fps)) *
                    min(cls().MAX_NUM_FRAMES, duration)
                )
            else:
                packing_nums = math.ceil(
                    duration * choose_fps / cls().MAX_NUM_FRAMES
                )
                if packing_nums <= cls().MAX_NUM_PACKING:
                    choose_frames_nums = round(duration * choose_fps)
                else:
                    choose_frames_nums = round(
                        cls().MAX_NUM_FRAMES * cls().MAX_NUM_PACKING
                        )
                    packing_nums = cls().MAX_NUM_PACKING
            if total_frames_num < choose_frames_nums :
                choose_frames_nums = total_frames_num
            frame_idx = cls().uniform_sample(
                total_frames_num, choose_frames_nums
            )
        else:
            full_read = num_frames == -1 or total_frames_num < num_frames
            if full_read:
                num_frames = total_frames_num
                frame_idx = list(range(0, num_frames))
            else:
                uniform_sampled_frames = np.linspace(
                    0, total_frames_num - 1, num_frames, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_idx), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(total_frames_num):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_idx:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        assert i == len(frame_idx), (f"Expected reading {len(frame_idx)} frames, "
                                 f"but only loaded {i} frames from video.")
        
        
        # Generate temporal_ids for each video
        # Extract fps from video metadata or use default
        frame_idx_ts = np.array(frame_idx) / original_fps
        scale = np.arange(0, duration, cls().TIME_SCALE)

        frame_ts_id = cls().map_to_nearest_scale(
            frame_idx_ts, scale
        ) / cls().TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)
        frame_ts_id_group = cls().group_array(frame_ts_id, packing_nums)
        frame_ts_id_group = [group.tolist() for group in frame_ts_id_group]
        
        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "enhanced_opencv",
            "temporal_ids": frame_ts_id_group
        }

        return frames, metadata


class VideoMediaIO(MediaIO[npt.NDArray]):
    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs
        video_loader_backend = envs.VLLM_VIDEO_LOADER_BACKEND
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in data.split(",")]
            ), {}

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
