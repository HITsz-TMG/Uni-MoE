#!/usr/bin/env python3
"""Validate a clone, checkpoint, and the pinned official Ascend runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

EXPECTED_ARCHITECTURE = "UniMoE2d5ForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_vl_div_moe"
EXPECTED_PROCESSOR = "UniMoE2d5Processor"
EXPECTED_VLLM = "0.22.1"
EXPECTED_VLLM_ASCEND = "0.22.1rc1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip imports that require the vLLM Ascend image",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def matches_expected_version(installed: str | None, expected: str) -> bool:
    """Match a public version while permitting a PEP 440 local build suffix."""
    if installed is None:
        return False
    try:
        return Version(installed).public == Version(expected).public
    except InvalidVersion:
        return False


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_path = args.model.expanduser().resolve()
    checks: list[dict[str, Any]] = []

    add_check(checks, "checkpoint.directory", model_path.is_dir(), str(model_path))
    config_path = model_path / "config.json"
    add_check(checks, "checkpoint.config_json", config_path.is_file(), str(config_path))

    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            add_check(checks, "checkpoint.config_parse", True, sha256(config_path))
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "checkpoint.config_parse", False, str(exc))

    architectures = config.get("architectures") or []
    add_check(
        checks,
        "checkpoint.architecture",
        EXPECTED_ARCHITECTURE in architectures,
        architectures,
    )
    add_check(
        checks,
        "checkpoint.model_type",
        config.get("model_type") == EXPECTED_MODEL_TYPE,
        config.get("model_type"),
    )
    add_check(
        checks,
        "checkpoint.tokenizer",
        (model_path / "tokenizer_config.json").is_file(),
        str(model_path / "tokenizer_config.json"),
    )
    processor_candidates = [
        model_path / "processor_config.json",
        model_path / "preprocessor_config.json",
    ]
    processor_metadata_paths = [path for path in processor_candidates if path.is_file()]
    add_check(
        checks,
        "checkpoint.processor_metadata",
        bool(processor_metadata_paths),
        [str(path) for path in processor_metadata_paths],
    )
    processor_metadata: dict[str, Any] = {}
    if processor_metadata_paths:
        try:
            processor_metadata = json.loads(
                processor_metadata_paths[0].read_text(encoding="utf-8")
            )
            add_check(
                checks,
                "checkpoint.processor_class",
                processor_metadata.get("processor_class") == EXPECTED_PROCESSOR,
                processor_metadata.get("processor_class"),
            )
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "checkpoint.processor_class", False, str(exc))
    add_check(
        checks,
        "repository.chat_templates",
        all(
            (repo_root / "assets" / name).is_file()
            for name in ("chat_template_hf.jinja", "chat_template_vllm.jinja")
        ),
        str(repo_root / "assets"),
    )

    if not args.static_only:
        vllm_version = package_version("vllm")
        ascend_version = package_version("vllm-ascend")
        add_check(
            checks,
            "runtime.vllm",
            matches_expected_version(vllm_version, EXPECTED_VLLM),
            vllm_version,
        )
        add_check(
            checks,
            "runtime.vllm_ascend",
            matches_expected_version(ascend_version, EXPECTED_VLLM_ASCEND),
            ascend_version,
        )

        entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
        matching = [ep.value for ep in entry_points if ep.name == "unimoe2d5"]
        add_check(checks, "runtime.plugin_entrypoint", bool(matching), matching)

        try:
            from unimoe2d5.plugin import register

            register()
            from vllm import ModelRegistry

            add_check(
                checks,
                "runtime.model_registry",
                EXPECTED_ARCHITECTURE in ModelRegistry.get_supported_archs(),
                EXPECTED_ARCHITECTURE,
            )
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "runtime.model_registry", False, repr(exc))

        hf_config = None
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            add_check(
                checks,
                "runtime.auto_config",
                getattr(hf_config, "model_type", None) == EXPECTED_MODEL_TYPE,
                type(hf_config).__name__,
            )
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "runtime.auto_config", False, repr(exc))

        try:
            from transformers import AutoProcessor

            if hf_config is None:
                raise RuntimeError("AutoConfig failed; processor resolution was not attempted")
            hf_processor = AutoProcessor.from_pretrained(
                model_path,
                config=hf_config,
                trust_remote_code=True,
            )
            add_check(
                checks,
                "runtime.auto_processor",
                type(hf_processor).__name__ == EXPECTED_PROCESSOR,
                f"{type(hf_processor).__module__}:{type(hf_processor).__name__}",
            )
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "runtime.auto_processor", False, repr(exc))

        try:
            from unimoe2d5.hf.modeling_unimoe2d5 import (
                UniMoE2d5ForConditionalGeneration as HFModel,
            )
            from unimoe2d5.vllm.model import (
                UniMoE2d5ForConditionalGeneration as VLLMModel,
            )

            add_check(
                checks,
                "runtime.model_imports",
                HFModel.__name__ == VLLMModel.__name__ == EXPECTED_ARCHITECTURE,
                [HFModel.__module__, VLLMModel.__module__],
            )
        except Exception as exc:  # noqa: BLE001
            add_check(checks, "runtime.model_imports", False, repr(exc))

    report = {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "model_path": str(model_path),
        "static_only": args.static_only,
        "status": "pass" if all(check["ok"] for check in checks) else "fail",
        "checks": checks,
    }
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            marker = "PASS" if check["ok"] else "FAIL"
            print(f"[{marker}] {check['name']}: {check['detail']}")
        print(f"preflight: {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
