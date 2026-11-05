"""Aggregate metrics_results.json files under results/ into a markdown table and a bar chart.

Usage:
    python scripts/summarize_results.py                  # prints tables for every SMI function
    python scripts/summarize_results.py --plot assets/results.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

LABELS = {
    "random_output": "Source Only (random)",
    "fl_rand_output": "FL-Random",
    "rand_dass_output": "Random-DASS",
    "dass_output": "DASS (ours)",
}
METRICS = ["Accuracy", "F1 Score", "AUC-ROC", "Average Precision"]


def collect(results_dir):
    """-> {smi_fn: {method: [metrics_dict, ...]}}"""
    table = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(results_dir, "*", "run*", "*_output", "results", "metrics_results.json")):
        parts = path.split(os.sep)
        smi_fn, method = parts[-5], parts[-3]
        with open(path) as f:
            table[smi_fn][method].append(json.load(f))
    return table


def to_markdown(smi_fn, by_method):
    n_runs = max(len(v) for v in by_method.values())
    lines = [f"**{smi_fn}** " + (f"(mean ± std over {n_runs} runs)" if n_runs > 1 else "(single run)"), "",
             "| Method | " + " | ".join(METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for method, label in LABELS.items():
        runs = by_method.get(method)
        if not runs:
            continue
        cells = []
        for m in METRICS:
            vals = np.array([r[m] for r in runs])
            cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) > 1 else f"{vals.mean():.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(by_method, out_path, title):
    import matplotlib.pyplot as plt

    methods = [m for m in LABELS if m in by_method]
    acc = [np.mean([r["Accuracy"] for r in by_method[m]]) for m in methods]
    auc = [np.mean([r["AUC-ROC"] for r in by_method[m]]) for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color="#4C78A8")
    b2 = ax.bar(x + 0.18, auc, 0.36, label="AUC-ROC", color="#F58518")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, [LABELS[m] for m in methods])
    ax.set_ylim(0.85, 0.98)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plot", default=None, help="Save a bar chart of the fl1mi results to this path")
    args = parser.parse_args()

    table = collect(args.results_dir)
    for smi_fn in sorted(table):
        print(to_markdown(smi_fn, table[smi_fn]), end="\n\n")

    if args.plot and "fl1mi" in table:
        plot(table["fl1mi"], args.plot, "Amazon → Yelp (budget 1000, 30% target, FL1MI, mean of 5 runs)")
