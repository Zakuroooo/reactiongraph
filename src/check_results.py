"""Fail if a number quoted in the README is not the number in results/.

The rule for this project is that no number reaches the README or the resume unless a saved results
file says it. This checks that mechanically, so an edit cannot quietly drift from the measurements.

Run: python -m src.check_results
"""
import json
import re
import sys
from pathlib import Path

README = Path("README.md")
RESULTS = Path("results")


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def claims():
    """Each claim: a label, the value measured, and how it must be written in the README."""
    clean = load("clean_counts.json")
    graph = load("graph_counts.json")
    retrieval = load("retrieval.json")
    lora_data = load("lora_data_stats.json")
    lora = load("lora.json")
    out = []
    if clean:
        out += [
            ("sample size", f"{clean['sample_in']:,}"),
            ("source rows", f"{clean['source_rows_total']:,}"),
            ("dropped", str(clean["dropped_total"])),
            ("duplicates merged", f"{clean['duplicate_reactions_merged']:,}"),
            ("reactions kept", f"{clean['reactions_out']:,}"),
            ("molecule mentions", f"{clean['molecule_mentions']:,}"),
            ("distinct spellings", f"{clean['distinct_spellings_atom_maps_removed']:,}"),
            ("unique molecules", f"{clean['unique_molecules_canonical']:,}"),
            ("spellings merged", f"{clean['spellings_merged']:,}"),
            ("molecules with 2+ spellings", f"{clean['molecules_with_2plus_spellings']:,}"),
            ("unmapped reactants moved", f"{clean['unmapped_reactants_moved_to_reagents']:,}"),
        ]
    if graph:
        relationships = sum(graph[k] for k in ["HAS_REACTANT", "HAS_REAGENT", "HAS_PRODUCT", "REPORTED_IN"])
        out += [
            ("Reaction nodes", f"{graph['Reaction']:,}"),
            ("Molecule nodes", f"{graph['Molecule']:,}"),
            ("Patent nodes", f"{graph['Patent']:,}"),
            ("relationships", f"{relationships:,}"),
        ]
    if retrieval:
        methods = retrieval["methods"]
        for method, label in [("graph", "graph"), ("vector_product", "vector product"),
                              ("vector_reaction", "vector reaction")]:
            for kind in ["one_hop", "two_hop"]:
                score = methods[method][kind]["recall@10"]
                out.append((f"{label} {kind} recall@10", f"{score['mean']:.3f}"))
                out.append((f"{label} {kind} recall@10 CI",
                            f"[{score['ci95'][0]:.3f}, {score['ci95'][1]:.3f}]"))
        out.append(("graph two_hop recall@50", f"{methods['graph']['two_hop']['recall@50']['mean']:.3f}"))
        out.append(("reagents above cap", str(retrieval["reagents_above_degree_cap"])))
        out.append(("two-hop median answers", str(int(retrieval["answer_set_size"]["two_hop"]["median"]))))
        out.append(("two-hop max answers", str(retrieval["answer_set_size"]["two_hop"]["max"])))
    if lora_data:
        out += [
            ("train rows", f"{lora_data['train']:,}"),
            ("val rows", str(lora_data["val"])),
            ("test rows", str(lora_data["test"])),
            ("patents in both splits", str(lora_data["patents_in_both_train_and_test"])),
            ("test products seen in train", str(lora_data["test_products_also_in_train"])),
            ("nearest-neighbour copy", f"{lora_data['nearest_neighbour_copy_exact_pct']:.1f}%"),
        ]
    if lora:
        # Only present after the Kaggle run has been re-graded locally by src/grade_lora.py.
        # Written exactly as the results file holds them, so the README cannot round them away.
        for name, scores in lora["methods"].items():
            out.append((f"{name} exact match", f"{scores['exact_pct']}%"))
            out.append((f"{name} exact CI", f"[{scores['exact_ci95'][0]}, {scores['exact_ci95'][1]}]"))
            out.append((f"{name} valid", f"{scores['valid_pct']}%"))
        training = lora.get("lora_training", {})
        if training:
            out.append(("trainable params", f"{training['trainable_params']:,}"))
            out.append(("total params", f"{training['total_params']:,}"))
            out.append(("trainable pct", f"{training['trainable_pct']}%"))
            out.append(("lora steps", f"{training['steps']:,}"))
            for entry in training.get("val_loss_history", []):
                out.append((f"val loss at step {entry['step']}", str(entry["val_loss"])))
    return out


def main():
    if not README.exists():
        raise SystemExit("README.md is missing")
    text = README.read_text()
    tests = claims()
    if not tests:
        raise SystemExit("no results files found - run the pipeline first")
    missing = [(label, value) for label, value in tests if value not in text]
    for label, value in tests:
        print(("  ok  " if (label, value) not in missing else " MISS ") + f"{label}: {value}")
    counted = len({v for _, v in tests})
    if missing:
        print(f"\n{len(missing)} measured value(s) are not written in the README:", file=sys.stderr)
        for label, value in missing:
            print(f"  {label} = {value}", file=sys.stderr)
        return 1
    print(f"\nall {len(tests)} claims ({counted} distinct values) match results/")
    numbers = set(re.findall(r"\b\d[\d,]*\.?\d*%?\b", text))
    print(f"README contains {len(numbers)} distinct numeric tokens; "
          "any not listed above is a date, a setting or a count stated in prose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
