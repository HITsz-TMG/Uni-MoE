#!/usr/bin/env python3
"""Small OpenAI-compatible client for text/image/audio/video requests."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--prompt", default="Describe the input.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def _data_url(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _resolve_model(base_url: str, requested: str) -> str:
    if requested != "auto":
        return requested
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    content: list[dict] = []
    content.extend(
        {"type": "image_url", "image_url": {"url": _data_url(path)}}
        for path in args.image
    )
    content.extend(
        {"type": "audio_url", "audio_url": {"url": _data_url(path)}}
        for path in args.audio
    )
    content.extend(
        {"type": "video_url", "video_url": {"url": _data_url(path)}}
        for path in args.video
    )
    content.append({"type": "text", "text": args.prompt})

    payload = {
        "model": _resolve_model(base_url, args.model),
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "fps": args.fps,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
