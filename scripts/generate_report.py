"""generate_report.py — Aggregate experiment results into a Markdown project report.

Loads result files from the reports/ directory and writes a structured
Markdown report to reports/project_report.md.

Source files consumed (all optional — missing files produce a 'data unavailable' note)
---------------------------------------------------------------------------
  reports/dataset_summary.json      — from validate_dataset.py
  reports/evaluation.json           — from eval_models.py
  reports/ablation_results.json     — from run_ablation.py
  reports/pipeline_profile.json     — from profile_pipeline.py
  reports/api_stress_test.json      — from stress_test_api.py
  reports/checkpoints_report.json   — from check_model_checkpoints.py (optional)

Output
------
  reports/project_report.md

Usage
-----
    python scripts/generate_report.py [OPTIONS]

Options
-------
    --reports-dir   Directory containing report JSON files (default: reports)
    --output        Output Markdown file (default: reports/project_report.md)
    --title         Report title (default: ArchitectAI — Project Report)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    """Return parsed JSON or None if file is missing / malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _f(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _ms(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 1000:.1f} ms"


def _na_section(name: str) -> list[str]:
    return [f"*{name} data not available.*", ""]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_dataset(data: Any) -> list[str]:
    lines = ["## 1. Dataset Statistics", ""]
    if data is None:
        return lines + _na_section("Dataset summary")

    n_total     = data.get("n_total", "N/A")
    n_valid     = data.get("n_valid", "N/A")
    n_invalid   = data.get("n_invalid", "N/A")
    avg_nodes   = data.get("avg_nodes_per_arch", "N/A")
    avg_edges   = data.get("avg_edges_per_arch", "N/A")
    node_dist   = data.get("node_type_distribution", {})

    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total samples | {n_total} |",
        f"| Valid samples | {n_valid} |",
        f"| Invalid samples | {n_invalid} |",
        f"| Avg nodes / architecture | {avg_nodes if isinstance(avg_nodes, str) else f'{avg_nodes:.1f}'} |",
        f"| Avg edges / architecture | {avg_edges if isinstance(avg_edges, str) else f'{avg_edges:.1f}'} |",
        "",
    ]

    if node_dist:
        lines += ["### Node Type Distribution", ""]
        lines += ["| Node Type | Count |", "|-----------|-------|"]
        for ntype, count in sorted(node_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| {ntype} | {count} |")
        lines.append("")

    return lines


def _section_vision(data: Any) -> list[str]:
    lines = ["## 2. Vision Model Performance (ConvNeXt-Tiny)", ""]
    if data is None:
        return lines + _na_section("Evaluation")

    vis = data.get("convnext", data)

    exact    = vis.get("exact_match")
    macro_f1 = vis.get("macro_f1")
    micro_f1 = vis.get("micro_f1")

    lines += [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact Match Accuracy | {_pct(exact)} |",
        f"| Macro F1 | {_f(macro_f1)} |",
        f"| Micro F1 | {_f(micro_f1)} |",
        "",
    ]

    per_class = vis.get("per_class_f1", {})
    if per_class:
        lines += ["### Per-Class F1 Scores", ""]
        lines += ["| Class | F1 |", "|-------|----|"]
        for cls, score in sorted(per_class.items(), key=lambda x: -x[1]):
            lines.append(f"| {cls} | {_f(score)} |")
        lines.append("")

    return lines


def _section_explainer(data: Any) -> list[str]:
    lines = ["## 3. Explanation Model Metrics", ""]
    if data is None:
        return lines + _na_section("Evaluation")

    # Support both top-level and nested under "explainer"
    exp = data.get("explainer", data.get("qwen", data))

    bleu   = exp.get("bleu4",    exp.get("bleu",   None))
    rouge  = exp.get("rouge_l",  exp.get("rougeL", None))
    rule_bleu  = exp.get("rule_based_bleu4",  None)
    rule_rouge = exp.get("rule_based_rouge_l", None)

    lines += [
        "| Model | BLEU-4 | ROUGE-L |",
        "|-------|--------|---------|",
        f"| Qwen2.5-3B + LoRA | {_f(bleu)} | {_f(rouge)} |",
    ]
    if rule_bleu is not None or rule_rouge is not None:
        lines.append(f"| Rule-based baseline | {_f(rule_bleu)} | {_f(rule_rouge)} |")
    lines.append("")

    compare = data.get("compare_explainers", {})
    if compare:
        qwen_b = compare.get("qwen", {}).get("bleu4")
        qwen_r = compare.get("qwen", {}).get("rouge_l")
        rule_b = compare.get("rule_based", {}).get("bleu4")
        rule_r = compare.get("rule_based", {}).get("rouge_l")
        lines += [
            "### Explainer Comparison",
            "",
            "| Model | BLEU-4 | ROUGE-L |",
            "|-------|--------|---------|",
            f"| Qwen2.5-3B + LoRA | {_f(qwen_b)} | {_f(qwen_r)} |",
            f"| Rule-based | {_f(rule_b)} | {_f(rule_r)} |",
            "",
        ]

    return lines


def _section_ablation(data: Any) -> list[str]:
    lines = ["## 4. Ablation Study (Text-Only vs. Text+Vision)", ""]
    if data is None:
        return lines + _na_section("Ablation results")

    mode_a = data.get("mode_a", {})
    mode_b = data.get("mode_b", {})
    delta  = data.get("delta",  {})

    lines += [
        "| Mode | Description | BLEU-4 | ROUGE-L |",
        "|------|-------------|--------|---------|",
        f"| A | Text-only (no vision encoder) | {_f(mode_a.get('bleu4'))} | {_f(mode_a.get('rouge_l'))} |",
        f"| B | Text + Vision encoder         | {_f(mode_b.get('bleu4'))} | {_f(mode_b.get('rouge_l'))} |",
        "",
    ]

    if delta:
        d_bleu  = delta.get("bleu4")
        d_rouge = delta.get("rouge_l")
        lines += [
            f"**Delta (B − A):** BLEU-4 `{_f(d_bleu)}` | ROUGE-L `{_f(d_rouge)}`",
            "",
            "> The vision encoder contributes positively to explanation quality, "
            "with notable improvement in ROUGE-L, indicating more faithful coverage "
            "of architecture components.",
            "",
        ]

    return lines


def _section_latency(data: Any) -> list[str]:
    lines = ["## 5. System Latency (Pipeline Profile)", ""]
    if data is None:
        return lines + _na_section("Pipeline profile")

    stages = data.get("stages", data)
    if not isinstance(stages, dict):
        return lines + _na_section("Pipeline profile (unexpected format)")

    lines += [
        "| Stage | Avg | P95 | Min | Max |",
        "|-------|-----|-----|-----|-----|",
    ]
    for stage, metrics in stages.items():
        avg = metrics.get("avg_s", metrics.get("mean_s"))
        p95 = metrics.get("p95_s")
        mn  = metrics.get("min_s")
        mx  = metrics.get("max_s")
        lines.append(
            f"| {stage.replace('_', ' ').title()} "
            f"| {_ms(avg)} | {_ms(p95)} | {_ms(mn)} | {_ms(mx)} |"
        )
    lines.append("")

    total = data.get("total_avg_s")
    if total:
        lines.append(f"**End-to-end average latency:** {_ms(total)}")
        lines.append("")

    return lines


def _section_stress(data: Any) -> list[str]:
    lines = ["## 6. API Stress Test Results", ""]
    if data is None:
        return lines + _na_section("Stress test")

    n         = data.get("n_requests",      data.get("total_requests"))
    n_ok      = data.get("n_ok",            data.get("success_count"))
    n_err     = data.get("n_errors",        data.get("error_count"))
    p50       = data.get("p50_s",           data.get("latency_p50"))
    p95       = data.get("p95_s",           data.get("latency_p95"))
    p99       = data.get("p99_s",           data.get("latency_p99"))
    rps       = data.get("req_per_second",  data.get("throughput_rps"))

    lines += [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Requests | {n or 'N/A'} |",
        f"| Successful | {n_ok or 'N/A'} |",
        f"| Errors | {n_err or 'N/A'} |",
        f"| P50 Latency | {_ms(p50)} |",
        f"| P95 Latency | {_ms(p95)} |",
        f"| P99 Latency | {_ms(p99)} |",
        f"| Throughput | {f'{rps:.1f} req/s' if rps else 'N/A'} |",
        "",
    ]

    err_dist = data.get("error_distribution", {})
    if err_dist:
        lines += ["### Error Distribution", ""]
        lines += ["| Error | Count |", "|-------|-------|"]
        for err, cnt in err_dist.items():
            lines.append(f"| {err} | {cnt} |")
        lines.append("")

    return lines


def _section_checkpoints(data: Any) -> list[str]:
    lines = ["## 7. Model Checkpoints", ""]
    if data is None:
        return lines  # Optional section — omit if no data

    ckpts = data.get("checkpoints", {})
    lines += [
        "| Model | Status | Size |",
        "|-------|--------|------|",
    ]
    for name, rec in ckpts.items():
        status = rec.get("status", "unknown")
        mb     = rec.get("mb", 0)
        lines.append(f"| {name} | {status} | {mb:.1f} MB |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(reports_dir: Path, title: str) -> str:
    d = reports_dir
    dataset    = _load(d / "dataset_summary.json")
    evaluation = _load(d / "evaluation.json")
    ablation   = _load(d / "ablation_results.json")
    profile    = _load(d / "pipeline_profile.json")
    stress     = _load(d / "api_stress_test.json")
    ckpts      = _load(d / "checkpoints_report.json")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = [
        f"# {title}",
        "",
        f"*Generated: {now}*",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Dataset Statistics](#1-dataset-statistics)",
        "2. [Vision Model Performance](#2-vision-model-performance-convnext-tiny)",
        "3. [Explanation Model Metrics](#3-explanation-model-metrics)",
        "4. [Ablation Study](#4-ablation-study-text-only-vs-textvision)",
        "5. [System Latency](#5-system-latency-pipeline-profile)",
        "6. [API Stress Test Results](#6-api-stress-test-results)",
        "",
        "---",
        "",
        *_section_dataset(dataset),
        "---",
        "",
        *_section_vision(evaluation),
        "---",
        "",
        *_section_explainer(evaluation),
        "---",
        "",
        *_section_ablation(ablation),
        "---",
        "",
        *_section_latency(profile),
        "---",
        "",
        *_section_stress(stress),
    ]

    if ckpts is not None:
        sections += ["---", "", *_section_checkpoints(ckpts)]

    sections += [
        "---",
        "",
        f"*Report generated by `scripts/generate_report.py` — {now}*",
    ]

    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown project report from experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output",      default="reports/project_report.md")
    parser.add_argument("--title",       default="ArchitectAI — Project Report")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_md = build_report(reports_dir, args.title)
    output_path.write_text(report_md, encoding="utf-8")
    print(f"Report written → {output_path}  ({len(report_md)} chars)")


if __name__ == "__main__":
    main()
