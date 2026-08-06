#!/usr/bin/env python3
"""Summarize authoritative model-side DSpark decode rates from a server log."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DECODE_PATTERN = re.compile(
    r"\[deepseek4\] DSpark decode: "
    r"(?P<tokens>\d+) tok in (?P<seconds>\d+(?:\.\d+)?)s "
    r"\((?P<rate>\d+(?:\.\d+)?) tok/s\) "
    r"accept_rate=(?P<acceptance>\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class DecodeRecord:
    tokens: int
    seconds: float
    tok_s: float
    acceptance: float


def parse_records(log_text: str) -> list[DecodeRecord]:
    return [
        DecodeRecord(
            tokens=int(match.group("tokens")),
            seconds=float(match.group("seconds")),
            tok_s=float(match.group("rate")),
            acceptance=float(match.group("acceptance")),
        )
        for match in DECODE_PATTERN.finditer(log_text)
    ]


def summarize_records(
    records: Sequence[DecodeRecord],
    targets: Sequence[int],
    warmup: int,
    runs: int,
    expected_tokens: int,
) -> dict[str, object]:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    requests_per_target = warmup + runs
    expected_records = len(targets) * requests_per_target
    if len(records) != expected_records:
        raise ValueError(
            f"found {len(records)} DSpark decode records; expected {expected_records}"
        )

    groups: list[dict[str, object]] = []
    offset = 0
    for target in targets:
        all_rows = list(records[offset : offset + requests_per_target])
        offset += requests_per_target
        if any(row.tokens != expected_tokens for row in all_rows):
            actual = sorted({row.tokens for row in all_rows})
            raise ValueError(
                f"target {target} completion lengths {actual}; expected {expected_tokens}"
            )
        measured = all_rows[warmup:]
        rates = [row.tok_s for row in measured]
        total_seconds = sum(row.seconds for row in measured)
        groups.append({
            "target_context": target,
            "n": len(measured),
            "completion_tokens": expected_tokens,
            "server_decode_tok_s_median": round(statistics.median(rates), 3),
            "server_decode_tok_s_min": round(min(rates), 3),
            "server_decode_tok_s_max": round(max(rates), 3),
            "server_decode_tok_s_weighted": round(
                sum(row.tokens for row in measured) / total_seconds, 3
            ),
            "acceptance_median": round(
                statistics.median(row.acceptance for row in measured), 4
            ),
            "records": [asdict(row) for row in measured],
        })
    return {
        "schema_version": 1,
        "metric": "model-side DSpark decode",
        "warmup": warmup,
        "runs": runs,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        payload = summarize_records(
            parse_records(args.server_log.read_text(encoding="utf-8")),
            args.targets,
            args.warmup,
            args.runs,
            args.expected_tokens,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2) + "\n"
    args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
