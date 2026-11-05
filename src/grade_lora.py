"""Re-grade the Kaggle run's saved predictions locally, so no number is taken on trust.

The notebook already grades as it goes, but it reports its own summary. This reads only the raw
prediction files, grades them again with src/grading.py, and writes results/lora.json.

Run: python -m src.grade_lora path/to/kaggle_outputs
"""
import json
import sys
from pathlib import Path

from src.grading import grade, mcnemar_exact, wilson

METHODS = ["base", "few_shot", "retrieval", "lora"]
OUT = Path("results/lora.json")


def grade_file(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    graded = {}
    disagreements = 0
    for row in rows:
        valid, exact = grade(row["reply"], row["product"])
        if "exact" in row and row["exact"] != exact:
            disagreements += 1
        graded[row["id"]] = {"valid": valid, "exact": exact}
    return graded, disagreements


def summarise(name, graded, seconds_per_answer=None):
    n = len(graded)
    n_valid = sum(g["valid"] for g in graded.values())
    n_exact = sum(g["exact"] for g in graded.values())
    out = {"n": n, "valid": n_valid, "exact": n_exact,
           "valid_pct": round(100 * n_valid / n, 2), "exact_pct": round(100 * n_exact / n, 2),
           "valid_ci95": [round(100 * x, 2) for x in wilson(n_valid, n)],
           "exact_ci95": [round(100 * x, 2) for x in wilson(n_exact, n)]}
    if seconds_per_answer is not None:
        out["seconds_per_answer"] = seconds_per_answer
    return out


def main(folder):
    folder = Path(folder)
    notebook = {}
    if (folder / "results.json").exists():
        notebook = json.loads((folder / "results.json").read_text())
    report = {"graded_from": str(folder), "regraded_here": True,
              "notebook_settings": notebook.get("settings", {}),
              "lora_training": notebook.get("lora", {}), "methods": {}, "disagreements_with_notebook": {}}
    exact = {}
    for name in METHODS:
        path = folder / f"predictions_{name}.jsonl"
        if not path.exists():
            print("missing", path, "- skipping", file=sys.stderr)
            continue
        graded, disagreements = grade_file(path)
        exact[name] = {i: g["exact"] for i, g in graded.items()}
        report["methods"][name] = summarise(
            name, graded, notebook.get("methods", {}).get(name, {}).get("seconds_per_answer"))
        report["disagreements_with_notebook"][name] = disagreements
    if not report["methods"]:
        raise SystemExit(f"no prediction files found in {folder}")
    pairwise = {}
    names = list(exact)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = set(exact[a]) & set(exact[b])
            only_a = sum(exact[a][i_] and not exact[b][i_] for i_ in ids)
            only_b = sum(exact[b][i_] and not exact[a][i_] for i_ in ids)
            pairwise[f"{a}_vs_{b}"] = {f"only_{a}": only_a, f"only_{b}": only_b,
                                       "p_value_mcnemar": round(mcnemar_exact(only_a, only_b), 4)}
    report["pairwise"] = pairwise
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (v["exact_pct"], v["exact_ci95"]) for k, v in report["methods"].items()}, indent=2))
    print(json.dumps(pairwise, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_outputs")
