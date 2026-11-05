"""Part 5 data: split reactions by patent, attach retrieved neighbours, write the notebook's input.

Whole patents go to one side, so near-copies from the same patent cannot leak into the test set.
Retrieval only searches the training set. Run: python -m src.lora_data
"""
import gzip
import json
import random
from pathlib import Path

from src import graph
from src.grading import wilson
from src.vectors import VectorIndex, ranked_answers

OUT = Path("data/lora/lora_data.json.gz")
STATS = Path("results/lora_data_stats.json")
SEED = 42
N_TEST = 500
N_VAL = 200
N_TRAIN = 20_000
N_EXAMPLES = 3
MAX_INPUT_CHARS = 300
MAX_PRODUCT_CHARS = 150


def usable(r):
    """One product, and short enough to fit the model's context on the free GPU."""
    inputs = ".".join(r["reactants"] + r["reagents"])
    return len(r["products"]) == 1 and len(inputs) <= MAX_INPUT_CHARS and len(r["products"][0]) <= MAX_PRODUCT_CHARS


def split_by_patent(records, seed=SEED, test_fraction=0.05, val_fraction=0.02):
    patents = sorted({p for r in records for p in r["patents"]})
    random.Random(seed).shuffle(patents)
    n_test = int(len(patents) * test_fraction)
    n_val = int(len(patents) * val_fraction)
    test_patents = set(patents[:n_test])
    val_patents = set(patents[n_test:n_test + n_val])
    train, val, test = [], [], []
    for r in records:
        mine = set(r["patents"])
        if mine <= test_patents:
            test.append(r)
        elif mine <= val_patents:
            val.append(r)
        elif not mine & (test_patents | val_patents):
            train.append(r)
        # A reaction reported in patents on two sides is left out, so nothing crosses the split.
    return train, val, test


def slim(r):
    return {"id": r["id"], "reactants": ".".join(r["reactants"]), "reagents": ".".join(r["reagents"]),
            "product": r["products"][0]}


def main():
    records = [r for r in graph.read_records() if usable(r)]
    train, val, test = split_by_patent(records)
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    available = {"train": len(train), "val": len(val), "test": len(test)}
    train, val, test = train[:N_TRAIN], val[:N_VAL], test[:N_TEST]

    index = VectorIndex(train, "inputs")
    rows = []
    nn_correct = 0
    for r in test:
        hits = index.search(r["reactants"] + r["reagents"], k=N_EXAMPLES)
        row = slim(r)
        row["neighbours"] = [slim(h) for h, _ in hits]
        row["neighbour_scores"] = [round(s, 4) for _, s in hits]
        nn_correct += ranked_answers(hits[:1], "products", k=1) == r["products"]
        rows.append(row)

    train_products = {p for r in train for p in r["products"]}
    seen = sum(r["products"][0] in train_products for r in test)
    data = {"train": [slim(r) for r in train], "val": [slim(r) for r in val], "test": rows,
            "few_shot": [slim(r) for r in rng.sample(train, N_EXAMPLES)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(json.dumps(data).encode())

    train_patents = {p for r in train for p in r["patents"]}
    test_patents = {p for r in test for p in r["patents"]}
    stats = {
        "usable_reactions": len(records), "available_after_split": available,
        "train": len(train), "val": len(val), "test": len(test), "seed": SEED,
        "patents_in_both_train_and_test": len(train_patents & test_patents),
        "test_products_also_in_train": seen,
        "test_products_also_in_train_pct": round(100 * seen / len(test), 1),
        "nearest_neighbour_copy_exact_pct": round(100 * nn_correct / len(test), 1),
        "nearest_neighbour_copy_ci95": [round(100 * x, 1) for x in wilson(nn_correct, len(test))],
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print("bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
