from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from unimoe2d5.ffmpeg import require_ffmpeg, resolve_ffmpeg


class FfmpegResolutionTests(unittest.TestCase):
    @patch("unimoe2d5.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_prefers_system_ffmpeg(self, _which) -> None:
        self.assertEqual(resolve_ffmpeg(), "/usr/bin/ffmpeg")

    @patch("unimoe2d5.ffmpeg.shutil.which", return_value=None)
    def test_uses_imageio_fallback(self, _which) -> None:
        module = ModuleType("imageio_ffmpeg")
        module.get_ffmpeg_exe = lambda: "/bundled/ffmpeg"  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"imageio_ffmpeg": module}):
            self.assertEqual(resolve_ffmpeg(), "/bundled/ffmpeg")

    @patch("unimoe2d5.ffmpeg.resolve_ffmpeg", return_value=None)
    def test_required_ffmpeg_has_actionable_error(self, _resolve) -> None:
        with self.assertRaisesRegex(RuntimeError, "demo/requirements.txt"):
            require_ffmpeg()
