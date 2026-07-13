from __future__ import annotations

import argparse
from itertools import islice, product
import json
from pathlib import Path
import statistics
import time
from typing import Any


RESULT_SCHEMA = {
    "kind": "microbench",
    "metric": "name of the measured quantity, for example latency_ms or bandwidth_gbps",
    "unit": "unit associated with metric",
    "aggregation": "median/p95/mean over repeat samples",
}


def run_microbench_plan(
    script: str,
    exp_id: str,
    purpose: str,
    sweep: dict[str, list[Any]],
    metrics: list[str],
    server_verify: str,
) -> int:
    parser = argparse.ArgumentParser(description=purpose)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="results/microbench.jsonl")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    script_name = Path(script).name
    payload = {
        "kind": "microbench",
        "script": script_name,
        "exp_id": exp_id,
        "purpose": purpose,
        "sweep": sweep,
        "metrics": metrics,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "output": args.output,
        "result_schema": RESULT_SCHEMA,
        "recommended_command": [
            "torchrun",
            "--nproc_per_node=${NPROC:-4}",
            str(Path("microbench") / script_name),
            "--output",
            args.output,
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
        ],
        "server_verify": True,
        "server_verify_detail": server_verify,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for index, case in enumerate(_iter_cases(sweep, args.max_cases)):
            for _ in range(args.warmup):
                _local_case_probe(case)
            samples_ms = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                _local_case_probe(case)
                samples_ms.append((time.perf_counter() - start) * 1000.0)
            latency = statistics.median(samples_ms) if samples_ms else 0.0
            row = {
                "kind": "microbench",
                "script": script_name,
                "exp_id": exp_id,
                "case_index": index,
                "case": case,
                "status": "SERVER-VERIFY",
                "server_verify_detail": server_verify,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "samples_ms": samples_ms,
                "metrics": {
                    "latency_ms": latency,
                },
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "rows": index + 1 if "index" in locals() else 0}, sort_keys=True))
    return 0


def _iter_cases(sweep: dict[str, list[Any]], max_cases: int) -> Any:
    keys = list(sweep)
    combos = (dict(zip(keys, values, strict=True)) for values in product(*(sweep[key] for key in keys)))
    if max_cases > 0:
        return islice(combos, max_cases)
    return combos


def _local_case_probe(case: dict[str, Any]) -> int:
    total = 0
    for key, value in case.items():
        total += len(str(key)) + len(str(value))
    return total
