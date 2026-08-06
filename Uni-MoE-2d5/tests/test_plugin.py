from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from unimoe2d5.plugin import _register_transformers_processor


class TransformersProcessorRegistrationTests(unittest.TestCase):
    def test_registers_installed_processor_idempotently(self):
        class FakeConfig:
            pass

        class FakeProcessor:
            pass

        calls = []

        class FakeAutoProcessor:
            @staticmethod
            def register(config_class, processor_class, *, exist_ok=False):
                calls.append((config_class, processor_class, exist_ok))

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = FakeAutoProcessor
        fake_config_module = types.ModuleType(
            "unimoe2d5.hf.configuration_unimoe2d5"
        )
        fake_config_module.UniMoE2d5Config = FakeConfig
        fake_processor_module = types.ModuleType(
            "unimoe2d5.hf.processing_unimoe2d5"
        )
        fake_processor_module.UniMoE2d5Processor = FakeProcessor

        modules = {
            "transformers": fake_transformers,
            "unimoe2d5.hf.configuration_unimoe2d5": fake_config_module,
            "unimoe2d5.hf.processing_unimoe2d5": fake_processor_module,
        }
        with patch.dict(sys.modules, modules):
            _register_transformers_processor()
            _register_transformers_processor()

        self.assertEqual(
            calls,
            [
                (FakeConfig, FakeProcessor, True),
                (FakeConfig, FakeProcessor, True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
