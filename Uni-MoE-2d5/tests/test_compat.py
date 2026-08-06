from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from unimoe2d5.compat import (
    install_tokenizer_special_token_compat,
    install_vllm_mm_gather_compat,
)


class TokenizerCompatTests(unittest.TestCase):
    def test_only_unimoe_audio_list_is_remapped(self):
        class FakeTokenizerBase:
            def _set_model_specific_special_tokens(self, special_tokens):
                return special_tokens

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.tokenization_utils_base = types.SimpleNamespace(
            PreTrainedTokenizerBase=FakeTokenizerBase
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            self.assertTrue(install_tokenizer_special_token_compat())
            tokenizer = FakeTokenizerBase()
            audio = [
                "<|audio_start|>",
                "<|audio_end|>",
                "<|audio_pad|>",
                "<|audio_asr|>",
                "<|audio_caption|>",
            ]
            mapped = tokenizer._set_model_specific_special_tokens(audio)
            self.assertEqual(mapped["audio_token"], "<|audio_pad|>")
            unrelated = ["<image>", "<video>"]
            self.assertIs(tokenizer._set_model_specific_special_tokens(unrelated), unrelated)
            self.assertFalse(install_tokenizer_special_token_compat())


class VllmGatherCompatTests(unittest.TestCase):
    def test_only_missing_is_mm_embed_is_suppressed(self):
        class FakeRunner:
            mode = "missing"

            def _gather_mm_embeddings(self):
                if self.mode == "missing":
                    raise AttributeError("FakeRunner has no attribute 'is_mm_embed'")
                if self.mode == "other":
                    raise AttributeError("unrelated attribute failure")
                return [1], 2

        fake_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")
        fake_module.GPUModelRunner = FakeRunner
        with patch.dict(sys.modules, {"vllm.v1.worker.gpu_model_runner": fake_module}):
            self.assertTrue(install_vllm_mm_gather_compat())
            runner = FakeRunner()
            self.assertEqual(runner._gather_mm_embeddings(), ([], None))
            runner.mode = "ok"
            self.assertEqual(runner._gather_mm_embeddings(), ([1], 2))
            runner.mode = "other"
            with self.assertRaisesRegex(AttributeError, "unrelated"):
                runner._gather_mm_embeddings()


if __name__ == "__main__":
    unittest.main()
