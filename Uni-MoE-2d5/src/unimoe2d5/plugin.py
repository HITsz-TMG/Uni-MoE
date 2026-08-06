"""vLLM general-plugin entry point for UniMoE-2.5."""

from __future__ import annotations

import logging
import os

from .compat import (
    install_tokenizer_special_token_compat,
    install_vllm_mm_gather_compat,
)

ARCHITECTURE = "UniMoE2d5ForConditionalGeneration"
MODEL_CLASS = "unimoe2d5.vllm.model:UniMoE2d5ForConditionalGeneration"
MODEL_TYPE = "qwen3_vl_div_moe"

logger = logging.getLogger(__name__)


def _register_transformers_config() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from .hf.configuration_unimoe2d5 import UniMoE2d5Config

    try:
        existing = CONFIG_MAPPING[MODEL_TYPE]
    except KeyError:
        CONFIG_MAPPING.register(MODEL_TYPE, UniMoE2d5Config)
        return

    if existing is not UniMoE2d5Config:
        raise RuntimeError(
            f"Transformers config key {MODEL_TYPE!r} is already registered to "
            f"{existing.__module__}:{existing.__name__}; refusing an unsafe override."
        )


def _register_transformers_processor() -> None:
    """Bind the installed processor to the custom config class.

    Released checkpoints only name ``UniMoE2d5Processor`` in
    ``processor_config.json``; they do not carry an ``auto_map`` or a copy of
    the Python source. Registering the pair lets ``AutoProcessor`` resolve the
    installed out-of-tree implementation without checkpoint-side remote code.
    """
    from transformers import AutoProcessor

    from .hf.configuration_unimoe2d5 import UniMoE2d5Config
    from .hf.processing_unimoe2d5 import UniMoE2d5Processor

    # ``register`` is public in every Transformers version accepted by vLLM
    # 0.22.1. ``exist_ok`` makes repeated general-plugin discovery idempotent;
    # the config class itself is ours and is guarded above by MODEL_TYPE.
    AutoProcessor.register(
        UniMoE2d5Config,
        UniMoE2d5Processor,
        exist_ok=True,
    )


def _register_vllm_model() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS)


def register() -> None:
    """Register config, model, and narrowly scoped runtime compatibility."""
    install_tokenizer_special_token_compat()
    _register_transformers_config()
    _register_transformers_processor()
    _register_vllm_model()

    if os.environ.get("UNIMOE2D5_DISABLE_MM_GATHER_COMPAT", "0") != "1":
        install_vllm_mm_gather_compat()

    logger.info(
        "Registered config, processor, and %s via the UniMoE-2.5 out-of-tree plugin",
        ARCHITECTURE,
    )


__all__ = ["ARCHITECTURE", "MODEL_CLASS", "MODEL_TYPE", "register"]
