#!/usr/bin/env python3
"""Run UniMoE-2.5 with Hugging Face Transformers on an Ascend NPU."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local checkpoint directory")
    parser.add_argument("--prompt", default="Introduce yourself briefly.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def _content_blocks(args: argparse.Namespace) -> list[dict]:
    content: list[dict] = []
    content.extend({"type": "image", "image": path} for path in args.image)
    content.extend({"type": "audio", "audio": path} for path in args.audio)
    content.extend(
        {"type": "video", "video": path, "fps": args.fps}
        for path in args.video
    )
    content.append({"type": "text", "text": args.prompt})
    return content


def main() -> None:
    args = parse_args()

    import torch
    import torch_npu  # noqa: F401

    from unimoe2d5.compat import install_tokenizer_special_token_compat
    from unimoe2d5.hf.modeling_unimoe2d5 import (
        UniMoE2d5ForConditionalGeneration,
    )
    from unimoe2d5.hf.processing_unimoe2d5 import UniMoE2d5Processor
    from unimoe2d5.media import process_omni_info

    install_tokenizer_special_token_compat()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {model_path}")

    processor = UniMoE2d5Processor.from_pretrained(model_path)
    if not processor.chat_template:
        template_path = Path(__file__).resolve().parents[1] / "assets" / "chat_template_hf.jinja"
        processor.chat_template = template_path.read_text(encoding="utf-8")

    model = UniMoE2d5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.eval().to(args.device)

    messages = [{"role": "user", "content": _content_blocks(args)}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos, audios, video_kwargs = process_omni_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=True)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[prompt],
        images=images,
        videos=videos,
        audios=audios,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    if "audio_features" in inputs:
        inputs["audio_features"] = inputs["audio_features"].unsqueeze(0)
        inputs["audio_features_mask"] = inputs["audio_features_mask"].unsqueeze(0)
    inputs = inputs.to(args.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    continuation = generated_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(output)


if __name__ == "__main__":
    main()
