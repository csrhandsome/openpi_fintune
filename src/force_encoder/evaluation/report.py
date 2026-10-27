"""Create machine-readable and Markdown reports from held-out predictions.

Example::

    python -m force_encoder.evaluation.report --runs outputs/jepa_seed0 \
        outputs/jepa_seed1 --baseline outputs/random_seed0 --output reports/test.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .artifacts import load_evaluation_artifact
from .statistics import analyze_run


def build_report(
    runs: list[Path],
    *,
    baseline: Path | None = None,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 0,
    seed: int = 0,
) -> dict:
    baseline_artifact = load_evaluation_artifact(baseline) if baseline is not None else None
    entries = []
    families: dict[tuple, list[dict]] = defaultdict(list)
    seen_paths: set[Path] = set()
    for run in runs:
        artifact = load_evaluation_artifact(run)
        canonical = artifact.path.resolve()
        if canonical in seen_paths:
            raise ValueError(f"Duplicate run artifact: {run}")
        seen_paths.add(canonical)
        manifest = artifact.manifest
        # A single reference run cannot represent independent paired seed trials.
        # Only equal, explicitly recorded seeds qualify for a paired comparison.
        paired = (
            baseline_artifact is not None
            and manifest.get("seed") is not None
            and manifest.get("seed") == baseline_artifact.manifest.get("seed")
        )
        analysis = analyze_run(
            artifact,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples if paired else 0,
            seed=seed,
            baseline=baseline_artifact if paired else None,
        )
        entry = {"run": str(run), "manifest": manifest, **analysis}
        if baseline_artifact is not None and not paired:
            entry["comparison_note"] = (
                "Paired baseline comparison omitted: a baseline with the same explicit training seed is required."
            )
        entries.append(entry)
        family = manifest.get("run_family", manifest.get("model_name"))
        # Without a named family, a collection of different heads/methods is not
        # interpreted as repeat training seeds of a single method.
        if family is not None:
            family_configuration = {
                name: manifest.get(name)
                for name in (
                    "head",
                    "head_training",
                    "checkpoint_sha256",
                    "baseline",
                    "encoder_config",
                    "random_seed",
                    "target_key",
                    "units",
                    "original_class_ids",
                )
            }
            canonical_protocol = {
                name: sorted(value) if name.endswith("_group_ids") else value
                for name, value in manifest["protocol"].items()
            }
            key = (
                family,
                manifest["task"],
                manifest["task_type"],
                tuple(manifest["label_names"]),
                manifest["dataset_fingerprint"],
                manifest["split_fingerprint"],
                json.dumps(canonical_protocol, sort_keys=True),
                json.dumps(family_configuration, sort_keys=True),
            )
            families[key].append(entry)
    summaries = []
    for key, members in families.items():
        seeds = [member["manifest"].get("seed") for member in members]
        if len(members) < 2:
            continue
        if None in seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"Family {key[0]} has missing/duplicate seeds; cannot summarize seed variability")
        scalar_names = [name for name, value in members[0]["metrics"].items() if not isinstance(value, list)]
        summary = {"run_family": key[0], "task": key[1], "seeds": seeds, "num_runs": len(members), "metrics": {}}
        for name in scalar_names:
            values = [member["metrics"][name] for member in members]
            summary["metrics"][name] = (
                None
                if any(value is None for value in values)
                else {"mean": float(np.mean(values)), "std_across_seeds": float(np.std(values, ddof=1))}
            )
        summaries.append(summary)
    # Holm controls the family-wise error across every comparison metric/run
    # actually included in this report, without a scipy dependency.
    comparisons = [
        comparison
        for entry in entries
        for comparison in entry.get("comparison", {}).values()
        if comparison.get("pvalue") is not None
    ]
    ordered = sorted(comparisons, key=lambda item: item["pvalue"])
    previous = 0.0
    for index, comparison in enumerate(ordered):
        adjusted = min(1.0, max(previous, comparison["pvalue"] * (len(ordered) - index)))
        comparison["pvalue_holm"] = adjusted
        previous = adjusted
    return {
        "schema_version": 1,
        "runs": entries,
        "seed_summaries": summaries,
        "baseline": str(baseline) if baseline is not None else None,
        "notes": [
            "Metrics use original label units. Regression overall R2 averages only nonconstant label dimensions; undefined dimensions are null.",
            "Confidence intervals resample whole source/episode groups, retaining all cameras and frames in each selected group.",
            "Bootstrap uncertainty is test-episode uncertainty. Seed summaries use independent runs and do not pool frames as training trials.",
            "pvalue_holm corrects across all reported nonnull paired p-values using Holm family-wise correction.",
            "Macro F1 includes every declared class; balanced accuracy averages recalls of classes present in the test labels.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Force encoder downstream evaluation",
        "",
        "| Run | Episodes | Frames | Metric | Estimate | 95% cluster CI |",
        "|---|---:|---:|---|---:|---|",
    ]
    for entry in report["runs"]:
        for metric, interval in entry["confidence_intervals"].items():
            value = entry["metrics"][metric]
            estimate = "undefined" if value is None else f"{value:.6g}"
            bounds = "unavailable" if interval is None else f"[{interval[0]:.6g}, {interval[1]:.6g}]"
            lines.append(
                f"| {entry['run']} | {entry['num_groups']} | {entry['num_examples']} | {metric} | {estimate} | {bounds} |"
            )
        if entry.get("comparison"):
            lines.extend(
                [
                    "",
                    f"Paired comparison for `{entry['run']}` (candidate minus baseline):",
                    "",
                    "| Metric | Delta | Cluster CI | Group randomization p | Holm p |",
                    "|---|---:|---|---:|---:|",
                ]
            )
            for metric, comparison in entry["comparison"].items():
                delta, interval, pvalue = comparison["delta"], comparison["delta_ci"], comparison["pvalue"]
                lines.append(
                    f"| {metric} | {delta if delta is not None else 'undefined'} | {interval if interval is not None else 'unavailable'} | {pvalue if pvalue is not None else 'not computed'} | {comparison.get('pvalue_holm', 'not computed')} |"
                )
    if report["seed_summaries"]:
        lines.extend(["", "## Independent training seeds", ""])
        lines.extend(
            f"- {summary['run_family']} ({summary['num_runs']} seeds): {json.dumps(summary['metrics'])}"
            for summary in report["seed_summaries"]
        )
    lines.extend(["", *report["notes"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation_report.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = build_report(
        args.runs,
        baseline=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Saved {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
