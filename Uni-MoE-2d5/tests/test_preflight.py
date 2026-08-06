from __future__ import annotations

import unittest

from scripts.preflight import matches_expected_version


class RuntimeVersionTests(unittest.TestCase):
    def test_accepts_exact_and_local_build_versions(self) -> None:
        self.assertTrue(matches_expected_version("0.22.1", "0.22.1"))
        self.assertTrue(matches_expected_version("0.22.1+empty", "0.22.1"))
        self.assertTrue(matches_expected_version("0.22.1rc1+ascend", "0.22.1rc1"))

    def test_rejects_different_or_invalid_public_versions(self) -> None:
        self.assertFalse(matches_expected_version("0.22.1rc1", "0.22.1"))
        self.assertFalse(matches_expected_version("0.22.2", "0.22.1"))
        self.assertFalse(matches_expected_version("not-a-version", "0.22.1"))
        self.assertFalse(matches_expected_version(None, "0.22.1"))


if __name__ == "__main__":
    unittest.main()
