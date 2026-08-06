import base64
import os
import tempfile
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import requests
import torch

from .vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)


def _resample_audio_numpy(
    waveform: np.ndarray, original_sampling_rate: int, target_sampling_rate: int
) -> np.ndarray:
    """Resample with numpy interpolation to avoid worker deadlocks in some torchaudio+fork setups."""
    if original_sampling_rate == target_sampling_rate:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    if waveform.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    out_len = max(
        1, int(math.floor((waveform.shape[0] * float(target_sampling_rate)) / float(original_sampling_rate)))
    )
    src_idx = np.arange(waveform.shape[0], dtype=np.float64)
    dst_idx = np.linspace(0.0, waveform.shape[0] - 1.0, num=out_len, endpoint=True, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, waveform.astype(np.float64, copy=False))
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def _resample_audio(waveform: np.ndarray, original_sampling_rate: int, target_sampling_rate: int) -> np.ndarray:
    if original_sampling_rate == target_sampling_rate:
        return waveform

    # Default to numpy in dataloader workers for better fork-safety.
    # Set UNIMOE_AUDIO_RESAMPLE_BACKEND=torchaudio to force torchaudio.
    backend = os.environ.get("UNIMOE_AUDIO_RESAMPLE_BACKEND", "numpy").strip().lower()
    if backend != "torchaudio":
        return _resample_audio_numpy(waveform, original_sampling_rate, target_sampling_rate)

    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "torchaudio is required for audio resampling when UNIMOE_AUDIO_RESAMPLE_BACKEND=torchaudio."
        ) from e
    waveform_t = torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0)
    waveform = (
        torchaudio.functional.resample(waveform_t, original_sampling_rate, target_sampling_rate)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return waveform


def _decode_audio_from_path_or_bytes(audio: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("soundfile is required for loading audio files.") from e

    payload: Optional[bytes] = None
    audio_path: Optional[str] = None
    temp_path: Optional[str] = None
    suffix = ".tmp"

    if audio.startswith("http://") or audio.startswith("https://"):
        with requests.get(audio, stream=True) as response:
            response.raise_for_status()
            payload = response.content
        parsed = urlparse(audio)
        ext = os.path.splitext(parsed.path)[1]
        if ext:
            suffix = ext
    elif audio.startswith("data:audio"):
        if "base64," not in audio:
            raise ValueError("Unsupported audio data URI, expected base64 payload.")
        _, base64_data = audio.split("base64,", 1)
        payload = base64.b64decode(base64_data)
    else:
        audio_path = audio[7:] if audio.startswith("file://") else audio
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        ext = os.path.splitext(audio_path)[1]
        if ext:
            suffix = ext

    def _decode_with_torchaudio(local_path: str) -> Tuple[np.ndarray, int]:
        try:
            import torchaudio
        except ImportError as e:
            raise ImportError(
                "torchaudio is required to decode non-audio containers (e.g. mp4 used as audio input)."
            ) from e
        waveform_t, sampling_rate_t = torchaudio.load(local_path)
        waveform_t = waveform_t.float()
        if waveform_t.ndim == 2:
            waveform_t = waveform_t.mean(dim=0)
        elif waveform_t.ndim != 1:
            waveform_t = waveform_t.reshape(-1)
        return waveform_t.cpu().numpy().astype(np.float32, copy=False), int(sampling_rate_t)

    try:
        if payload is not None:
            with BytesIO(payload) as bio:
                waveform, sampling_rate = sf.read(bio, dtype="float32", always_2d=False)
        else:
            waveform, sampling_rate = sf.read(audio_path, dtype="float32", always_2d=False)  # type: ignore[arg-type]
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, int(sampling_rate)
    except Exception as sf_exc:
        try:
            if payload is not None:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(payload)
                    temp_path = tmp.name
                return _decode_with_torchaudio(temp_path)
            return _decode_with_torchaudio(audio_path)  # type: ignore[arg-type]
        except Exception as ta_exc:
            raise RuntimeError(
                f"Failed to decode audio input with both soundfile and torchaudio. "
                f"soundfile_error={sf_exc}; torchaudio_error={ta_exc}"
            ) from ta_exc
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)


def fetch_audio(
    audio: Any,
    sampling_rate: int = 16000,
    max_seconds: float = 120.0,
) -> np.ndarray:
    """Read and preprocess a single audio input into a mono float32 waveform."""
    if isinstance(audio, dict):
        audio_source = audio.get("audio", audio.get("audio_url"))
        if audio_source is None:
            raise ValueError("Audio dict should contain `audio` or `audio_url`.")
        sampling_rate = int(audio.get("sampling_rate", sampling_rate))
        max_seconds = float(audio.get("max_seconds", max_seconds))
        audio = audio_source

    if isinstance(audio, np.ndarray):
        waveform = audio.astype(np.float32, copy=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
    elif torch.is_tensor(audio):
        waveform = audio.detach().cpu().float().numpy()
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
    elif isinstance(audio, str):
        waveform, original_sr = _decode_audio_from_path_or_bytes(audio)
        waveform = _resample_audio(waveform, original_sr, sampling_rate)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    max_samples = int(max_seconds * sampling_rate)
    if max_samples > 0 and waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    return np.ascontiguousarray(waveform, dtype=np.float32)


def split_audio_chunks(
    waveform: np.ndarray,
    sampling_rate: int = 16000,
    window_seconds: float = 30.0,
) -> list[np.ndarray]:
    window_samples = int(sampling_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"`window_seconds` must be positive, got {window_seconds}.")
    chunks = [waveform[i : i + window_samples] for i in range(0, waveform.shape[0], window_samples)]
    chunks = [chunk for chunk in chunks if chunk.shape[0] > 0]
    if len(chunks) == 0:
        chunks = [np.zeros((window_samples,), dtype=np.float32)]
    return [np.ascontiguousarray(chunk, dtype=np.float32) for chunk in chunks]


def preprocess_audio_chunks(
    audio: Any,
    sampling_rate: int = 16000,
    window_seconds: float = 30.0,
    max_seconds: float = 6000.0,
) -> list[np.ndarray]:
    waveform = fetch_audio(audio, sampling_rate=sampling_rate, max_seconds=max_seconds)
    return split_audio_chunks(waveform, sampling_rate=sampling_rate, window_seconds=window_seconds)


def extract_audio_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    audio_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "audio" in ele
                        or "audio_url" in ele
                        or ele.get("type", "text") in ("audio", "audio_url", "sound")
                    ):
                        audio_infos.append(ele)
    return audio_infos


def process_omni_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
    audio_sampling_rate: int = 16000,
    audio_window_seconds: float = 30.0,
    audio_max_seconds: float = 120.0,
) -> Union[
    Tuple[
        Optional[List[Any]],
        Optional[List[Union[torch.Tensor, List[Any]]]],
        Optional[List[list[np.ndarray]]],
    ],
    Tuple[
        Optional[List[Any]],
        Optional[List[Union[torch.Tensor, List[Any]]]],
        Optional[List[list[np.ndarray]]],
        Dict[str, Any],
    ],
]:
    vision_outputs = process_vision_info(
        conversations,
        return_video_kwargs=return_video_kwargs,
        return_video_metadata=return_video_metadata,
        image_patch_size=image_patch_size,
    )
    if return_video_kwargs:
        image_inputs, video_inputs, video_kwargs = vision_outputs
    else:
        image_inputs, video_inputs = vision_outputs
        video_kwargs = None

    audio_infos = extract_audio_info(conversations)
    audio_inputs = [
        preprocess_audio_chunks(
            audio_info,
            sampling_rate=audio_sampling_rate,
            window_seconds=audio_window_seconds,
            max_seconds=audio_max_seconds,
        )
        for audio_info in audio_infos
    ]
    if len(audio_inputs) == 0:
        audio_inputs = None

    if return_video_kwargs:
        return image_inputs, video_inputs, audio_inputs, video_kwargs
    return image_inputs, video_inputs, audio_inputs


__all__ = [
    "smart_resize",
    "fetch_image",
    "fetch_video",
    "extract_vision_info",
    "process_vision_info",
    "fetch_audio",
    "split_audio_chunks",
    "preprocess_audio_chunks",
    "extract_audio_info",
    "process_omni_info",
]
