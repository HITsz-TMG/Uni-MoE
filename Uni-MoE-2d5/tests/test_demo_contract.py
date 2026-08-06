from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

try:
    import httpx

    from demo import app as demo_app
    from scripts.create_smoke_media import create_video
except (ImportError, RuntimeError):
    httpx = None
    demo_app = None


@unittest.skipIf(demo_app is None, "install the demo extra to run gateway tests")
class DemoContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.payloads = []

        async def upstream(request):
            if request.method == "GET" and request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "unimoe-test"}]})
            if request.method == "POST" and request.url.path == "/v1/chat/completions":
                self.payloads.append(json.loads(request.content))
                body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
                return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(404)

        self.upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="http://upstream"
        )
        demo_app.state["client"] = self.upstream
        demo_app.state["semaphore"] = asyncio.Semaphore(1)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=demo_app.app), base_url="http://demo"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.upstream.aclose()

    async def test_text_request_resolves_model_and_streams_sse(self):
        response = await self.client.post(
            "/api/chat",
            data={"text": "hello", "history": "[]", "max_tokens": "16"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(self.payloads[0]["model"], "unimoe-test")
        self.assertNotIn("fps", self.payloads[0])

    async def test_public_endpoints_do_not_expose_upstream_details(self):
        config = await self.client.get("/api/config")
        self.assertEqual(config.status_code, 200)
        self.assertNotIn("model", config.json())
        self.assertNotIn("base_url", config.json())

        health = await self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"ok": True})

    async def test_video_request_uses_vllm_video_block_and_fps(self):
        previous = demo_app.SPLIT_AUDIO_FROM_VIDEO
        demo_app.SPLIT_AUDIO_FROM_VIDEO = False
        try:
            response = await self.client.post(
                "/api/chat",
                data={"text": "describe", "history": "[]", "fps": "3"},
                files={"files": ("clip.mp4", b"test-video", "video/mp4")},
            )
        finally:
            demo_app.SPLIT_AUDIO_FROM_VIDEO = previous
        self.assertEqual(response.status_code, 200)
        content = self.payloads[0]["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "video_url")
        self.assertTrue(content[0]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual(self.payloads[0]["fps"], 3.0)

    async def test_video_audio_track_is_sent_as_seekable_wav(self):
        extracted = b"RIFF-seekable-wave"
        with patch.object(
            demo_app,
            "_audio_from_video",
            AsyncMock(return_value=extracted),
        ) as split_audio:
            response = await self.client.post(
                "/api/chat",
                data={"text": "describe", "history": "[]", "fps": "2"},
                files={"files": ("clip.mp4", b"video-with-audio", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        split_audio.assert_awaited_once_with(b"video-with-audio", "clip.mp4")
        content = self.payloads[0]["messages"][-1]["content"]
        expected = base64.b64encode(extracted).decode("ascii")
        self.assertEqual(content[0]["type"], "audio_url")
        self.assertEqual(
            content[0]["audio_url"]["url"],
            f"data:audio/wav;base64,{expected}",
        )
        self.assertEqual(content[1]["type"], "video_url")

    @unittest.skipIf(
        demo_app is None or demo_app.FFMPEG is None,
        "install the demo ffmpeg dependency to run video audio extraction",
    )
    async def test_video_audio_extractor_writes_seekable_16khz_mono_wav(self):
        previous = demo_app.SPLIT_AUDIO_FROM_VIDEO
        demo_app.SPLIT_AUDIO_FROM_VIDEO = True
        try:
            with tempfile.TemporaryDirectory() as directory:
                video_path = Path(directory) / "test.mp4"
                create_video(video_path, demo_app.FFMPEG)
                extracted = await demo_app._audio_from_video(
                    video_path.read_bytes(), video_path.name
                )
        finally:
            demo_app.SPLIT_AUDIO_FROM_VIDEO = previous

        self.assertIsNotNone(extracted)
        with wave.open(io.BytesIO(extracted), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 16000)
            self.assertGreater(audio.getnframes(), 0)

    async def test_audio_request_is_normalized_to_wav(self):
        normalized = b"RIFF-normalized-wave"
        with patch.object(
            demo_app,
            "_normalize_audio",
            AsyncMock(return_value=normalized),
        ) as normalize:
            response = await self.client.post(
                "/api/chat",
                data={"text": "transcribe", "history": "[]"},
                files={"files": ("recording.m4a", b"input-container", "audio/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        normalize.assert_awaited_once_with(
            b"input-container", "recording.m4a", "audio/mp4"
        )
        content = self.payloads[0]["messages"][-1]["content"]
        expected = base64.b64encode(normalized).decode("ascii")
        self.assertEqual(content[0]["type"], "audio_url")
        self.assertEqual(
            content[0]["audio_url"]["url"],
            f"data:audio/wav;base64,{expected}",
        )

    async def test_audio_normalizer_writes_seekable_16khz_mono_pcm_wav(self):
        source = io.BytesIO()
        with wave.open(source, "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00\x00\x00" * 800)

        normalized = await demo_app._normalize_audio(
            source.getvalue(), "source.wav", "audio/wav"
        )

        with wave.open(io.BytesIO(normalized), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 16000)
            self.assertGreater(audio.getnframes(), 0)

    async def test_audio_decode_failure_is_reported_before_upstream(self):
        error = demo_app.HTTPException(status_code=422, detail="bad audio")
        with patch.object(
            demo_app,
            "_normalize_audio",
            AsyncMock(side_effect=error),
        ):
            response = await self.client.post(
                "/api/chat",
                data={"text": "transcribe", "history": "[]"},
                files={"files": ("broken.mp3", b"not-audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "bad audio")
        self.assertEqual(self.payloads, [])

    async def test_upload_limit_is_enforced_while_reading(self):
        previous = demo_app._LIMITS_MB["image"]
        demo_app._LIMITS_MB["image"] = 1 / (1024 * 1024)
        try:
            response = await self.client.post(
                "/api/chat",
                data={"text": "describe", "history": "[]"},
                files={"files": ("large.png", b"xx", "image/png")},
            )
        finally:
            demo_app._LIMITS_MB["image"] = previous
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.payloads, [])


if __name__ == "__main__":
    unittest.main()
