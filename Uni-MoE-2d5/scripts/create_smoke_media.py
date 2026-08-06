#!/usr/bin/env python3
"""Create small synthetic image, audio, and video files for runtime smoke tests."""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from unimoe2d5.ffmpeg import require_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/unimoe2d5-smoke-media"),
    )
    return parser.parse_args()


def create_image(path: Path) -> None:
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 290, 210), fill="royalblue")
    draw.ellipse((110, 70, 210, 170), fill="yellow")
    draw.text((85, 185), "UniMoE image test", fill="white")
    image.save(path)


def create_audio(path: Path, sample_rate: int = 16_000, duration: int = 2) -> None:
    samples = (
        int(0.25 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(sample_rate * duration)
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def create_video(path: Path, ffmpeg: str) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=16000",
            "-t",
            "2",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "test.png"
    audio_path = output_dir / "test.wav"
    video_path = output_dir / "test.mp4"
    create_image(image_path)
    create_audio(audio_path)
    create_video(video_path, require_ffmpeg())

    for path in (image_path, audio_path, video_path):
        print(f"{path}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
