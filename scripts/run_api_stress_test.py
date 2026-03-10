"""run_api_stress_test.py — API stress test runner for ArchitectAI.

Calls stress_test_api.py via subprocess with --n 100 --concurrency 5 (defaults),
then loads the JSON results and prints a formatted metrics table.

Usage
-----
    python scripts/run_api_stress_test.py [OPTIONS]

Options
-------
    --url         Base API URL             (default: http://localhost:8000)
    --n           Number of requests       (default: 100)
    --concurrency Concurrent workers       (default: 5)
    --output      Report path             (default: reports/api_stress_test.json)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SEP  = "=" * 60
_DASH = "-" * 60


def _run_stress(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/stress_test_api.py",
        "--url",         args.url,
        "--n",           str(args.n),
        "--concurrency", str(args.concurrency),
        "--output",      args.output,
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, text=True).returncode


def _ms(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 1000:.1f} ms"
    except (TypeError, ValueError):
        return str(val)


def _rps(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f} req/s"
    except (TypeError, ValueError):
        return str(val)


def _pct(n, total) -> str:
    if total is None or total == 0:
        return "N/A"
    try:
        return f"{int(n) / int(total) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _print_results(report: dict) -> None:
    n_req = report.get("n_requests",     report.get("total_requests"))
    n_ok  = report.get("n_ok",           report.get("success_count"))
    n_err = report.get("n_errors",       report.get("error_count"))
    p50   = report.get("p50_s",          report.get("latency_p50"))
    p95   = report.get("p95_s",          report.get("latency_p95"))
    p99   = report.get("p99_s",          report.get("latency_p99"))
    avg   = report.get("avg_s",          report.get("avg_latency"))
    rps   = report.get("req_per_second", report.get("throughput_rps"))

    print()
    print(_SEP)
    print("  API Stress Test Results")
    print(_SEP)
    print(f"  {'Total requests':<28} {n_req or 'N/A'}")
    print(f"  {'Successful':<28} {n_ok or 'N/A'}  ({_pct(n_ok, n_req)} success rate)")
    print(f"  {'Errors':<28} {n_err or 0}")
    print(_DASH)
    print(f"  {'Avg latency':<28} {_ms(avg)}")
    print(f"  {'P50 latency':<28} {_ms(p50)}")
    print(f"  {'P95 latency':<28} {_ms(p95)}")
    print(f"  {'P99 latency':<28} {_ms(p99)}")
    print(f"  {'Throughput':<28} {_rps(rps)}")

    err_dist: dict = report.get("error_distribution", {})
    if err_dist:
        print(_DASH)
        print("  Error Distribution")
        for err, cnt in err_dist.items():
            print(f"    {err:<40} {cnt}")

    print(_SEP)

    # Simple pass/fail judgement
    try:
        error_rate = int(n_err or 0) / int(n_req or 1)
        if error_rate == 0:
            print("  ✓ Zero errors — API healthy under load.")
        elif error_rate < 0.05:
            print(f"  ⚠ Error rate {error_rate:.1%} — investigate failures.")
        else:
            print(f"  ✗ High error rate {error_rate:.1%} — API may be overwhelmed.")
    except (TypeError, ValueError):
        pass
    print()


def main() -> None:
    args = _parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(_SEP)
    print("  ArchitectAI — API Stress Test")
    print(_SEP)
    print(f"  URL         : {args.url}")
    print(f"  Requests    : {args.n}")
    print(f"  Concurrency : {args.concurrency}")
    print(_SEP)

    rc = _run_stress(args)

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"ERROR: stress_test_api.py did not produce {out_path}")
        sys.exit(1)

    report = json.loads(out_path.read_text(encoding="utf-8"))
    _print_results(report)

    print(f"Report → {out_path}")
    sys.exit(rc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ArchitectAI API stress test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",         default="http://localhost:8000")
    p.add_argument("--n",           type=int, default=100)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--output",      default="reports/api_stress_test.json")
    return p.parse_args()


if __name__ == "__main__":
    main()
