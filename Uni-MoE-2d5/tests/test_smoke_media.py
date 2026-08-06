from __future__ import annotations

import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.create_smoke_media import create_audio, create_image, create_video
from unimoe2d5.ffmpeg import require_ffmpeg


class SmokeMediaTests(unittest.TestCase):
    def test_creates_expected_image_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "test.png"
            audio_path = Path(directory) / "test.wav"
            create_image(image_path)
            create_audio(audio_path)

            with Image.open(image_path) as image:
                self.assertEqual(image.size, (320, 240))
                self.assertEqual(image.mode, "RGB")
            with wave.open(str(audio_path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 16_000)
                self.assertEqual(audio.getnframes(), 32_000)

    @patch("scripts.create_smoke_media.subprocess.run")
    def test_video_generation_uses_resolved_ffmpeg(self, run) -> None:
        create_video(Path("test.mp4"), "/bundled/ffmpeg")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/bundled/ffmpeg")
        self.assertIn("testsrc=size=320x240:rate=2", command)
        self.assertIn("sine=frequency=660:sample_rate=16000", command)
        self.assertEqual(command[-1], "test.mp4")
        self.assertTrue(run.call_args.kwargs["check"])

    @unittest.skipUnless(
        importlib.util.find_spec("imageio_ffmpeg"),
        "install the demo extra to run the bundled ffmpeg integration test",
    )
    @patch("unimoe2d5.ffmpeg.shutil.which", return_value=None)
    def test_bundled_ffmpeg_creates_video(self, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            create_video(path, require_ffmpeg())
            self.assertGreater(path.stat().st_size, 0)
