from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in sorted(ROOT.rglob("*.py")):
            if "x_temp" in path.parts:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_plugin_entrypoint(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        entrypoints = data["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(entrypoints["unimoe2d5"], "unimoe2d5.plugin:register")

    def test_vllm_model_has_no_in_tree_relative_imports(self):
        source = (ROOT / "src/unimoe2d5/vllm/model.py").read_text(encoding="utf-8")
        self.assertNotIn("from .interfaces", source)
        self.assertNotIn("vllm.model_executor.models.unimoe2d5", source)

    def test_serve_script_does_not_expose_checkpoint_path_as_model_id(self):
        source = (ROOT / "scripts/serve_vllm.sh").read_text(encoding="utf-8")
        self.assertIn("--served-model-name", source)
        self.assertNotIn("model=${MODEL_PATH}", source)

    def test_release_tree_has_no_runtime_artifacts(self):
        forbidden_names = {"kernel_meta", "logs", "backup"}
        offenders = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.name in forbidden_names
            and not {".git", "x_temp"}.intersection(path.relative_to(ROOT).parts)
        ]
        self.assertEqual(offenders, [])

    def test_backend_specific_audio_templates(self):
        hf = (ROOT / "assets/chat_template_hf.jinja").read_text(encoding="utf-8")
        vllm = (ROOT / "assets/chat_template_vllm.jinja").read_text(encoding="utf-8")
        self.assertIn("<|audio_pad|>", hf)
        self.assertNotIn("<|audio_start|><|audio_pad|><|audio_end|>", hf)
        self.assertIn("<|audio_start|><|audio_pad|><|audio_end|>", vllm)

    def test_demo_config_example_is_not_secret_bearing(self):
        text = (ROOT / "demo/.env.example").read_text(encoding="utf-8")
        self.assertNotIn("TOKEN=", text.upper())
        self.assertNotIn("API_KEY=", text.upper())
        json.dumps({"env": text})


if __name__ == "__main__":
    unittest.main()
