"""Single-model web gateway for a UniMoE-2.5 vLLM server."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from unimoe2d5.ffmpeg import resolve_ffmpeg

HERE = Path(__file__).resolve().parent
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "auto")
API_KEY = os.environ.get("VLLM_API_KEY", "")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "2"))
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "8"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", "10"))
MAX_AUDIO_MB = float(os.environ.get("MAX_AUDIO_MB", "25"))
MAX_VIDEO_MB = float(os.environ.get("MAX_VIDEO_MB", "64"))
SPLIT_AUDIO_FROM_VIDEO = os.environ.get("SPLIT_AUDIO_FROM_VIDEO", "1") == "1"
AUDIO_FROM_VIDEO_SR = int(os.environ.get("AUDIO_FROM_VIDEO_SR", "16000"))
FFMPEG_TIMEOUT = float(os.environ.get("FFMPEG_TIMEOUT", "120"))
FFMPEG = resolve_ffmpeg()

logger = logging.getLogger("unimoe2d5.demo")

_LIMITS_MB = {
    "image": MAX_IMAGE_MB,
    "audio": MAX_AUDIO_MB,
    "video": MAX_VIDEO_MB,
}
_EXT_KIND = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".opus": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".m4v": "video",
}

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    state["client"] = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10),
        headers=headers,
    )
    state["semaphore"] = asyncio.Semaphore(MAX_CONCURRENCY)
    yield
    await state["client"].aclose()


app = FastAPI(title="UniMoE-2.5 Demo", lifespan=lifespan)


def _client() -> httpx.AsyncClient:
    return state["client"]  # type: ignore[return-value]


def _kind(upload: UploadFile) -> str | None:
    content_type = (upload.content_type or "").lower()
    for candidate in ("image", "audio", "video"):
        if content_type.startswith(f"{candidate}/"):
            return candidate
    return _EXT_KIND.get(Path(upload.filename or "").suffix.lower())


async def _read_upload(upload: UploadFile) -> tuple[str, str, str, bytes]:
    kind = _kind(upload)
    if kind is None:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    max_bytes = int(_LIMITS_MB[kind] * 1024 * 1024)
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"The {kind} upload exceeds the {_LIMITS_MB[kind]:g} MiB limit",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    media_type = upload.content_type or mimetypes.guess_type(upload.filename or "")[0]
    media_type = media_type or "application/octet-stream"
    return kind, media_type, upload.filename or f"upload.{kind}", data


def _content_block(kind: str, media_type: str, data: bytes) -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    block_type = f"{kind}_url"
    return {
        "type": block_type,
        block_type: {"url": f"data:{media_type};base64,{encoded}"},
    }


async def _normalize_audio(
    data: bytes,
    filename: str,
    declared_media_type: str,
    *,
    source_label: str = "audio upload",
) -> bytes:
    """Decode an uploaded audio container into a seekable 16 kHz mono PCM WAV."""
    if FFMPEG is None:
        raise HTTPException(
            status_code=503,
            detail="Audio normalization requires ffmpeg; install demo/requirements.txt",
        )
    if not data:
        raise HTTPException(status_code=422, detail="The audio upload is empty")

    logger.info(
        "normalizing %s declared_type=%r input_bytes=%d",
        source_label,
        declared_media_type,
        len(data),
    )
    with tempfile.TemporaryDirectory(prefix="unimoe2d5-audio-") as temp_dir:
        output_path = Path(temp_dir) / "normalized.wav"
        process = await asyncio.create_subprocess_exec(
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_FROM_VIDEO_SR),
            "-c:a",
            "pcm_s16le",
            str(output_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(input=data), timeout=FFMPEG_TIMEOUT
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            logger.warning("ffmpeg timed out while normalizing %s", source_label)
            raise HTTPException(
                status_code=422,
                detail="Timed out while decoding the audio upload",
            ) from exc

        if process.returncode != 0 or not output_path.is_file():
            detail = stderr.decode("utf-8", "replace").strip()
            logger.warning(
                "cannot decode %s: %s", source_label, detail[:300]
            )
            raise HTTPException(
                status_code=422,
                detail="Could not decode the audio upload",
            )

        normalized = output_path.read_bytes()

    if not normalized:
        raise HTTPException(status_code=422, detail="Decoded audio is empty")
    logger.info(
        "normalized %s output_type='audio/wav' output_bytes=%d",
        source_label,
        len(normalized),
    )
    return normalized


async def _audio_from_video(data: bytes, filename: str) -> bytes | None:
    """Extract a seekable 16 kHz mono PCM WAV from a video's audio track."""
    if not SPLIT_AUDIO_FROM_VIDEO:
        return None
    if FFMPEG is None:
        logger.warning("ffmpeg is unavailable; the embedded audio track will be skipped")
        return None
    try:
        return await _normalize_audio(
            data,
            filename,
            "video/embedded-audio",
            source_label="embedded video audio",
        )
    except HTTPException as exc:
        logger.info("no usable embedded audio track: %s", exc.detail)
        return None


def _clean_history(raw: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="history is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="history must be a JSON list")
    cleaned = []
    for message in parsed[-2 * HISTORY_TURNS :]:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            cleaned.append({"role": role, "content": content})
    return cleaned


async def _resolved_model() -> str:
    if MODEL != "auto":
        return MODEL
    response = await _client().get(f"{VLLM_BASE_URL}/models", timeout=10)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("vLLM returned an empty model list")
    return models[0]["id"]


def _sse_error(message: str) -> bytes:
    payload = json.dumps({"error": {"message": message}}, ensure_ascii=False)
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/config")
async def config() -> dict:
    return {
        "label": "UniMoE-2.5 6B",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "limits_mb": _LIMITS_MB,
    }


@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        await _resolved_model()
        return JSONResponse({"ok": True})
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "vLLM is unavailable"}, status_code=503)


@app.post("/api/chat")
async def chat(
    text: Annotated[str, Form()] = "",
    history: Annotated[str, Form()] = "[]",
    temperature: Annotated[float, Form()] = 0.0,
    max_tokens: Annotated[int, Form()] = 512,
    repetition_penalty: Annotated[float, Form()] = 1.05,
    frequency_penalty: Annotated[float, Form()] = 0.0,
    fps: Annotated[float, Form()] = 2.0,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> StreamingResponse:
    uploads = []
    for upload in files or []:
        uploads.append(await _read_upload(upload))

    explicit_audio = any(kind == "audio" for kind, _, _, _ in uploads)
    blocks_by_kind: dict[str, list[dict]] = {"image": [], "audio": [], "video": []}
    for kind, media_type, filename, data in uploads:
        if kind == "audio":
            normalized = await _normalize_audio(data, filename, media_type)
            blocks_by_kind["audio"].append(
                _content_block("audio", "audio/wav", normalized)
            )
        else:
            blocks_by_kind[kind].append(_content_block(kind, media_type, data))
        if kind == "video" and not explicit_audio:
            extracted = await _audio_from_video(data, filename)
            if extracted:
                blocks_by_kind["audio"].append(
                    _content_block("audio", "audio/wav", extracted)
                )
    blocks = (
        blocks_by_kind["image"]
        + blocks_by_kind["audio"]
        + blocks_by_kind["video"]
    )
    if not text.strip() and not blocks:
        raise HTTPException(status_code=400, detail="Enter text or attach media")
    if text.strip():
        blocks.append({"type": "text", "text": text.strip()})

    messages: list[dict] = _clean_history(history)
    messages.append({"role": "user", "content": blocks})
    try:
        served_model = await _resolved_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Cannot reach the vLLM service") from exc

    payload = {
        "model": served_model,
        "messages": messages,
        "stream": True,
        "temperature": min(max(float(temperature), 0.0), 2.0),
        "max_tokens": min(max(int(max_tokens), 1), MAX_OUTPUT_TOKENS),
        "repetition_penalty": min(max(float(repetition_penalty), 0.01), 2.0),
        "frequency_penalty": min(max(float(frequency_penalty), -2.0), 2.0),
    }
    if blocks_by_kind["video"]:
        payload["fps"] = min(max(float(fps), 0.1), 30.0)

    async def stream():
        semaphore: asyncio.Semaphore = state["semaphore"]  # type: ignore[assignment]
        async with semaphore:
            try:
                async with _client().stream(
                    "POST",
                    f"{VLLM_BASE_URL}/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        yield _sse_error(
                            f"vLLM request failed with status {response.status_code}"
                        )
                        return
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
            except httpx.RequestError:
                yield _sse_error("Request to vLLM failed")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
