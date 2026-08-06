"""Narrow compatibility shims for the pinned UniMoE-2.5 runtime."""

from __future__ import annotations

from typing import Any

_AUDIO_SPECIAL_TOKENS = (
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<|audio_asr|>",
    "<|audio_caption|>",
)
_AUDIO_SPECIAL_TOKEN_KEYS = (
    "audio_start_token",
    "audio_end_token",
    "audio_token",
    "audio_asr_token",
    "audio_caption_token",
)


def _token_text(token: Any) -> str:
    return str(getattr(token, "content", token))


def install_tokenizer_special_token_compat() -> bool:
    """Map UniMoE's legacy audio-token list without affecting other models."""
    from transformers import tokenization_utils_base

    owner = tokenization_utils_base.PreTrainedTokenizerBase
    current = owner._set_model_specific_special_tokens
    if getattr(current, "_unimoe2d5_compat", False):
        return False

    def patched(self, special_tokens):
        if (
            isinstance(special_tokens, list)
            and tuple(_token_text(token) for token in special_tokens[:5])
            == _AUDIO_SPECIAL_TOKENS
        ):
            special_tokens = {
                (
                    _AUDIO_SPECIAL_TOKEN_KEYS[index]
                    if index < len(_AUDIO_SPECIAL_TOKEN_KEYS)
                    else f"extra_token_{index}"
                ): token
                for index, token in enumerate(special_tokens)
            }
        return current(self, special_tokens)

    patched._unimoe2d5_compat = True
    patched._unimoe2d5_original = current
    owner._set_model_specific_special_tokens = patched
    return True


def install_vllm_mm_gather_compat() -> bool:
    """Handle the known pre-initialization ``is_mm_embed`` access on vLLM."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    current = GPUModelRunner._gather_mm_embeddings
    if getattr(current, "_unimoe2d5_compat", False):
        return False

    def patched(self, *args, **kwargs):
        try:
            return current(self, *args, **kwargs)
        except AttributeError as exc:
            if "is_mm_embed" not in str(exc):
                raise
            return [], None

    patched._unimoe2d5_compat = True
    patched._unimoe2d5_original = current
    GPUModelRunner._gather_mm_embeddings = patched
    return True


__all__ = [
    "install_tokenizer_special_token_compat",
    "install_vllm_mm_gather_compat",
]
