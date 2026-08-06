from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_ds4_server_decode.py"
SPEC = importlib.util.spec_from_file_location("summarize_ds4_server_decode", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ServerDecodeSummaryTest(unittest.TestCase):
    def test_groups_warmups_and_measured_records(self) -> None:
        rows = []
        for rate in (50.0, 70.0, 72.0, 40.0, 60.0, 64.0):
            rows.append(
                "[deepseek4] DSpark decode: 128 tok in 2.000s "
                f"({rate:.1f} tok/s) accept_rate=0.96"
            )
        records = MODULE.parse_records("\n".join(rows))
        # Repeated contexts are distinct sweep legs (for example the final 2K
        # request after a 16K burn-in) and must not overwrite one another.
        payload = MODULE.summarize_records(records, [2048, 2048], 1, 2, 128)
        groups = payload["groups"]
        self.assertEqual(groups[0]["target_context"], 2048)
        self.assertEqual(groups[0]["server_decode_tok_s_median"], 71.0)
        self.assertEqual(groups[1]["target_context"], 2048)
        self.assertEqual(groups[1]["server_decode_tok_s_median"], 62.0)
        self.assertEqual(groups[0]["n"], 2)

    def test_rejects_missing_or_short_records(self) -> None:
        record = MODULE.DecodeRecord(64, 1.0, 64.0, 1.0)
        with self.assertRaisesRegex(ValueError, "found 1.*expected 2"):
            MODULE.summarize_records([record], [2048], 0, 2, 64)
        with self.assertRaisesRegex(ValueError, "completion lengths"):
            MODULE.summarize_records([record], [2048], 0, 1, 128)


if __name__ == "__main__":
    unittest.main()
