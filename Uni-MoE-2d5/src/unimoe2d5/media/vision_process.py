import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


MAX_RATIO = 400
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 64
IMAGE_MAX_TOKEN_NUM = 2048
# Global token budget for each whole video (all frames), not per-frame budget.
VIDEO_MIN_TOKEN_NUM = 64
VIDEO_MAX_TOKEN_NUM = 4800

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 8
FPS_MAX_FRAMES = 150
MAX_NUM_WORKERS_FETCH_VIDEO = 8
DECORD_RETRY_NUM_THREADS = int(os.environ.get('DECORD_RETRY_NUM_THREADS', 1))
DECORD_SLOW_RETRY_MAX_FILE_SIZE_MB = int(os.environ.get('DECORD_SLOW_RETRY_MAX_FILE_SIZE_MB', 200))
DECORD_SLOW_RETRY_MAX_DURATION_SECONDS = float(os.environ.get('DECORD_SLOW_RETRY_MAX_DURATION_SECONDS', 300))

MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
logger = logging.getLogger(__name__)
VIDEO_READ_STATUS_CACHE: Dict[str, Dict[str, Any]] = {}


class VideoDecodeError(RuntimeError):
    pass


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = FRAME_FACTOR,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t_bar * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t_bar * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _uniform_sample_indices(start_frame: int, end_frame: int, nframes: int) -> List[int]:
    return torch.linspace(start_frame, end_frame, nframes).round().long().tolist()


def sample_video_frame_indices(
    start_frame: int,
    end_frame: int,
    nframes: int,
    video_fps: Union[int, float],
) -> List[int]:
    """Build frame indices for temporal patch size 2.

    Pattern (1-based positions):
    - odd positions use second-aligned anchors (multiples of rounded fps);
    - even positions are averages of neighboring odd positions;
    - the last even position is extrapolated by equal interval.
    """
    if nframes % FRAME_FACTOR != 0:
        logger.debug(
            f"sample_video_frame_indices fallback to uniform: nframes[{nframes}] is not divisible by FRAME_FACTOR[{FRAME_FACTOR}]."
        )
        return _uniform_sample_indices(start_frame, end_frame, nframes)
    if start_frame >= end_frame:
        logger.warning(
            f"sample_video_frame_indices fallback to uniform: invalid range [{start_frame}, {end_frame}]."
        )
        return _uniform_sample_indices(start_frame, end_frame, nframes)

    anchor_step = max(1, int(round(float(video_fps))))
    pair_num = nframes // FRAME_FACTOR
    first_anchor = math.ceil(start_frame / anchor_step) * anchor_step
    last_anchor = math.floor(end_frame / anchor_step) * anchor_step
    needed_anchors = pair_num

    if first_anchor > last_anchor:
        logger.debug(
            "sample_video_frame_indices fallback to uniform: "
            f"no anchor in range [{start_frame}, {end_frame}] for anchor_step[{anchor_step}]."
        )
        return _uniform_sample_indices(start_frame, end_frame, nframes)

    available_anchors = (last_anchor - first_anchor) // anchor_step + 1
    if available_anchors < needed_anchors:
        logger.debug(
            "sample_video_frame_indices fallback to uniform: "
            f"available anchors[{available_anchors}] < needed anchors[{needed_anchors}] "
            f"in range [{start_frame}, {end_frame}] with anchor_step[{anchor_step}]."
        )
        return _uniform_sample_indices(start_frame, end_frame, nframes)

    anchor_positions = torch.linspace(0, available_anchors - 1, needed_anchors).round().long().tolist()
    if len(set(anchor_positions)) != len(anchor_positions):
        logger.warning(
            "sample_video_frame_indices fallback to uniform: duplicated anchor positions after linspace rounding."
        )
        return _uniform_sample_indices(start_frame, end_frame, nframes)

    anchors = [first_anchor + pos * anchor_step for pos in anchor_positions]
    sampled_indices: List[int] = []
    for i in range(pair_num):
        left = anchors[i]
        if i < pair_num - 1:
            right = anchors[i + 1]
            middle = (left + right) // 2
        else:
            prev_left = anchors[i - 1] if i > 0 else max(start_frame, left - anchor_step)
            gap = max(1, left - prev_left)
            middle = min(end_frame, left + gap // 2)
        sampled_indices.extend([left, middle])

    return sampled_indices


def _read_video_torchvision(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(
        f"torchvision: {total_frames=}, {video_fps=}, "
        f"time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.tensor(
        sample_video_frame_indices(0, total_frames - 1, nframes, video_fps),
        dtype=torch.long,
    )
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _create_decord_video_reader(video_path: str, num_threads: Optional[int] = None):
    import decord

    kwargs = {}
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    return decord.VideoReader(video_path, **kwargs)


def _decord_batch_to_tensor(video_batch: Any) -> torch.Tensor:
    video = video_batch.asnumpy()
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def _normalize_video_cache_key(video_path: str) -> str:
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    return os.path.abspath(video_path)


def _get_video_file_size(video_path: str) -> Optional[int]:
    normalized_video_path = _normalize_video_cache_key(video_path)
    try:
        return os.path.getsize(normalized_video_path)
    except OSError:
        return None


def _update_video_read_status(video_path: str, status: str, error: Optional[str] = None):
    cache_key = _normalize_video_cache_key(video_path)
    VIDEO_READ_STATUS_CACHE[cache_key] = {
        "status": status,
        "size_bytes": _get_video_file_size(cache_key),
        "last_error": error,
    }


def _get_video_read_status(video_path: str) -> Optional[str]:
    cache_key = _normalize_video_cache_key(video_path)
    state = VIDEO_READ_STATUS_CACHE.get(cache_key)
    if state is None:
        return None
    return state.get("status")


def _raise_video_decode_error(_video_path: str, message: str):
    raise VideoDecodeError(message)


def _read_video_decord_once(
    ele: Dict[str, Any],
    num_threads: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    video_path = ele["video"]
    st = time.time()
    vr = _create_decord_video_reader(video_path, num_threads=num_threads)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # idx = sample_video_frame_indices(start_frame, end_frame, nframes, video_fps)
    # video = _decord_batch_to_tensor(vr.get_batch(idx))
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    logger.info(
        f"decord: {total_frames=}, {video_fps=}, "
        f"time={time.time() - st:.3f}s"
    )
    return video, video_metadata, sample_fps


def _get_video_duration_seconds(video_metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not video_metadata:
        return None
    fps = video_metadata.get("fps")
    total_num_frames = video_metadata.get("total_num_frames")
    if fps is None or total_num_frames is None:
        return None
    if fps <= 0:
        return None
    return float(total_num_frames) / float(fps)


def _probe_video_decord_metadata(ele: Dict[str, Any]) -> Dict[str, Any]:
    video_path = ele["video"]
    vr = _create_decord_video_reader(video_path, num_threads=None)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    if total_frames <= 0 or video_fps <= 0:
        raise ValueError(
            f"invalid video metadata for {video_path}: total_frames={total_frames}, video_fps={video_fps}"
        )
    return {
        "fps": video_fps,
        "total_num_frames": total_frames,
        "duration_seconds": float(total_frames) / float(video_fps),
    }


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    cached_status = _get_video_read_status(video_path)
    if cached_status == "bad":
        _raise_video_decode_error(video_path, "is marked as bad in video cache")

    if cached_status == "slow_ok":
        try:
            video, video_metadata, sample_fps = _read_video_decord_once(
                ele,
                num_threads=DECORD_RETRY_NUM_THREADS,
            )
            _update_video_read_status(video_path, "slow_ok")
            return video, video_metadata, sample_fps
        except Exception as slow_error:
            _update_video_read_status(
                video_path, "bad", error=type(slow_error).__name__
            )
            _raise_video_decode_error(
                video_path,
                "cached slow-path failed",
            )

    try:
        video, video_metadata, sample_fps = _read_video_decord_once(ele, num_threads=None)
        _update_video_read_status(video_path, "fast_ok")
        return video, video_metadata, sample_fps
    except Exception as fast_path_error:
        logger.warning(
            "decord fast-path failed; retrying with conservative reader settings; "
            "error_type=%s",
            type(fast_path_error).__name__,
        )
        file_size = _get_video_file_size(video_path)
        max_file_size_bytes = DECORD_SLOW_RETRY_MAX_FILE_SIZE_MB * 1024 * 1024
        duration_seconds = None
        try:
            duration_seconds = _probe_video_decord_metadata(ele).get("duration_seconds")
        except Exception as probe_error:
            logger.warning(
                "decord metadata probe failed after fast-path error; error_type=%s",
                type(probe_error).__name__,
            )
        if file_size is not None and file_size > max_file_size_bytes:
            _update_video_read_status(
                video_path, "bad", error=type(fast_path_error).__name__
            )
            _raise_video_decode_error(
                video_path,
                f"fast-path failed and file_size={file_size} exceeds "
                f"slow-retry limit {max_file_size_bytes}",
            )
        if duration_seconds is not None and duration_seconds > DECORD_SLOW_RETRY_MAX_DURATION_SECONDS:
            _update_video_read_status(
                video_path, "bad", error=type(fast_path_error).__name__
            )
            _raise_video_decode_error(
                video_path,
                f"fast-path failed and duration_seconds={duration_seconds:.3f} exceeds "
                f"slow-retry limit {DECORD_SLOW_RETRY_MAX_DURATION_SECONDS}",
            )
        try:
            video, video_metadata, sample_fps = _read_video_decord_once(
                ele,
                num_threads=DECORD_RETRY_NUM_THREADS,
            )
            _update_video_read_status(video_path, "slow_ok")
            return video, video_metadata, sample_fps
        except Exception as retry_error:
            _update_video_read_status(
                video_path, "bad", error=type(retry_error).__name__
            )
            _raise_video_decode_error(
                video_path,
                "slow-path failed",
            )


def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


def _read_video_torchcodec(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = sample_video_frame_indices(start_frame, end_frame, nframes, video_fps)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(
        f"torchcodec: {total_frames=}, {video_fps=}, "
        f"time={time.time() - st:.3f}s"
    )

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec",
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    return video_reader_backend


def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_min_total_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_max_total_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as exc:
            if video_reader_backend == "decord":
                raise
            logger.warning(
                "video_reader_backend %s failed; using torchvision; error_type=%s",
                video_reader_backend,
                type(exc).__name__,
            )
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
    else:
        # The input is a list of frames
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        # use ThreadPoolExecutor to parallel process frames
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element, **process_info}, image_factor)
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack([
            torch.from_numpy(np.array(image).transpose(2, 0, 1))
            for image in image_list
        ])

        # fake video metadata
        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
        )

    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", video_min_total_pixels)
    if min_pixels is None:
        min_pixels = video_min_total_pixels
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        max_pixels_limit = min(video_max_total_pixels, total_pixels)
        max_pixels_supposed = ele.get("max_pixels", max_pixels_limit)
        if max_pixels_supposed is None:
            max_pixels_supposed = max_pixels_limit
        if max_pixels_supposed > max_pixels_limit:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels_limit}].")
        max_pixels = min(max_pixels_supposed, max_pixels_limit)
        if min_pixels > max_pixels:
            logger.warning(f"The given min_pixels[{min_pixels}] exceeds max_pixels[{max_pixels}], clamp to max_pixels.")
            min_pixels = max_pixels
        resized_height, resized_width = smart_resize_video(
            nframes,
            height,
            width,
            temporal_factor=FRAME_FACTOR,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def extract_vision_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True,
                        image_patch_size=image_patch_size, return_video_metadata=return_video_metadata)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {'do_sample_frames': False}
    if not return_video_metadata: # BC for qwen2.5vl
        video_kwargs.update({'fps': video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
