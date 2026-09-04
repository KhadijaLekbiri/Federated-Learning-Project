"""CSV output and concise comparison reporting."""
from __future__ import annotations
import csv
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean
from experiment_types import ExperimentResult, RoundResult


REQUIRED_COLUMNS = [field.name for field in fields(RoundResult)]


def summarize(results: list[ExperimentResult]) -> list[dict]:
    rows = []
    for result in results:
        rounds = result.rounds
        attacked = [r for r in rounds if r.attack_mode != "honest" and r.round_id in result.config.attack_rounds]
        honest = [r for r in rounds if r.attack_mode == "honest" or r.round_id not in result.config.attack_rounds]
        recon = [r.dlg_reconstruction_error for r in rounds if r.dlg_reconstruction_error is not None]
        rows.append({"scenario_name": result.config.scenario_name,
            "attack_block_rate": mean([r.attack_detected or r.round_aborted for r in attacked]) if attacked else 0.0,
            "honest_false_positive_rate": mean([r.attack_detected for r in honest]) if honest else 0.0,
            "mean_reconstruction_error": mean(recon) if recon else None,
            "mean_f1": mean(r.f1 for r in rounds), "mean_round_time_ms": mean(r.round_runtime_ms for r in rounds),
            "aborted_round_rate": mean(r.round_aborted for r in rounds)})
    return rows


def write_results(results: list[ExperimentResult], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "round_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS); writer.writeheader()
        for result in results:
            for row in result.rounds: writer.writerow(asdict(row))
    summary = summarize(results)
    with (output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return summary


def console_table(summary: list[dict]) -> str:
    headers = ["scenario_name", "attack_block_rate", "honest_false_positive_rate", "mean_reconstruction_error", "mean_f1", "mean_round_time_ms", "aborted_round_rate"]
    formatted = [[str(row[h]) if not isinstance(row[h], float) else f"{row[h]:.4f}" for h in headers] for row in summary]
    widths = [max(len(h), *(len(row[i]) for row in formatted)) for i, h in enumerate(headers)]
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), "-+-".join("-"*w for w in widths)]
    lines += [" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in formatted]
    return "\n".join(lines)
