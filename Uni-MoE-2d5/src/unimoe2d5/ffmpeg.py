"""Resolve an ffmpeg executable without requiring a modified base image."""

from __future__ import annotations

import shutil


def resolve_ffmpeg() -> str | None:
    """Prefer a system ffmpeg, then the binary bundled by imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None

    try:
        return get_ffmpeg_exe()
    except RuntimeError:
        return None


def require_ffmpeg() -> str:
    """Return an ffmpeg executable or explain how to install the fallback."""
    executable = resolve_ffmpeg()
    if executable is None:
        raise RuntimeError(
            "ffmpeg is unavailable; install demo/requirements.txt to add the "
            "imageio-ffmpeg fallback"
        )
    return executable
