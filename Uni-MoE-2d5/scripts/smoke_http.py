#!/usr/bin/env python3
"""Text smoke test for an already-running UniMoE-2.5 vLLM server."""

from __future__ import annotations

import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--prompt", default="Reply with exactly: UniMoE ready")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if model == "auto":
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": 32,
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not content.strip():
        raise SystemExit("Smoke request returned an empty response")
    print(content)


if __name__ == "__main__":
    main()
