"""Part 3: graph retrieval vs vector search, scored by recall@k on programmatic questions.

The correct answers are computed from the cleaned data in plain Python. BIAS: those answers follow
the same links the graph stores, so this test favours the graph. See the README.

Run: python -m src.retrieval_eval   (saves results/retrieval.json and results/retrieval_questions.jsonl)
"""
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src import graph
from src.vectors import VectorIndex, ranked_answers

N_QUESTIONS = 200
MAX_REAGENT_DEGREE = 50  # above this a "reagent" is usually a common solvent or base
KS = [1, 5, 10, 20, 50]
SEARCH_DEPTH = 200  # how many reactions the vector search reads answers from
SEED = 7
OUT = Path("results/retrieval.json")
QUESTIONS = Path("results/retrieval_questions.jsonl")


def build_questions(records, n=N_QUESTIONS, seed=SEED, max_degree=MAX_REAGENT_DEGREE):
    """Return (one_hop, two_hop) question lists, each answer set computed directly from the records."""
    made_by = defaultdict(list)
    used_in = defaultdict(list)
    for r in records:
        for p in r["products"]:
            made_by[p].append(r)
        for g in r["reagents"]:
            used_in[g].append(r)
    degree = {g: len(rs) for g, rs in used_in.items()}
    rng = random.Random(seed)
    products = sorted(made_by)
    rng.shuffle(products)

    one_hop, two_hop = [], []
    for x in products:
        reagents = {g for r in made_by[x] for g in r["reagents"]}
        if reagents and len(one_hop) < n:
            one_hop.append({"type": "one_hop", "molecule": x, "answers": sorted(reagents)})
        rare = {g for g in reagents if degree[g] <= max_degree}
        others = {p for g in rare for r in used_in[g] for p in r["products"]} - {x}
        if others and len(two_hop) < n:
            two_hop.append({"type": "two_hop", "molecule": x, "answers": sorted(others)})
        if len(one_hop) == n and len(two_hop) == n:
            break
    return one_hop, two_hop


def recall_at(ranked, answers, k):
    return len(set(ranked[:k]) & set(answers)) / len(answers)


def score(rows, question_type, ks=KS, seed=SEED):
    """Mean recall@k with a bootstrap 95% CI over questions."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        values = np.array([recall_at(r["ranked"], r["answers"], k) for r in rows])
        boots = rng.choice(values, size=(2000, len(values))).mean(axis=1)
        out[f"recall@{k}"] = {"mean": round(float(values.mean()), 4),
                              "ci95": [round(float(np.quantile(boots, q)), 4) for q in (0.025, 0.975)]}
    out["n"] = len(rows)
    out["median_latency_ms"] = round(statistics.median(r["ms"] for r in rows), 2)
    out["type"] = question_type
    return out


def run_graph(driver, questions, max_k=max(KS)):
    rows = []
    for q in questions:
        start = time.perf_counter()
        if q["type"] == "one_hop":
            found = graph.query(driver, graph.REAGENTS_FOR, smiles=q["molecule"], limit=max_k)
        else:
            found = graph.query(driver, graph.OTHER_PRODUCTS, smiles=q["molecule"],
                                max_degree=MAX_REAGENT_DEGREE, limit=max_k)
        ms = (time.perf_counter() - start) * 1000
        rows.append({"ranked": [f["smiles"] for f in found], "answers": q["answers"], "ms": ms})
    return rows


def run_vector(index, questions, max_k=max(KS)):
    rows = []
    for q in questions:
        start = time.perf_counter()
        hits = index.search([q["molecule"]], k=SEARCH_DEPTH)
        if q["type"] == "one_hop":
            ranked = ranked_answers(hits, "reagents", k=max_k)
        else:
            ranked = ranked_answers(hits, "products", exclude={q["molecule"]}, k=max_k)
        ms = (time.perf_counter() - start) * 1000
        rows.append({"ranked": ranked, "answers": q["answers"], "ms": ms})
    return rows


def main():
    records = graph.read_records()
    one_hop, two_hop = build_questions(records)
    with QUESTIONS.open("w") as f:
        for q in one_hop + two_hop:
            f.write(json.dumps(q) + "\n")
    print("questions", len(one_hop), len(two_hop), flush=True)

    driver = graph.connect()
    report = {"settings": {"n_per_type": N_QUESTIONS, "max_reagent_degree_two_hop": MAX_REAGENT_DEGREE,
                           "vector_search_depth": SEARCH_DEPTH, "seed": SEED, "reactions": len(records)},
              "answer_set_size": {}, "methods": {}}
    for name, questions in [("one_hop", one_hop), ("two_hop", two_hop)]:
        sizes = [len(q["answers"]) for q in questions]
        report["answer_set_size"][name] = {"median": statistics.median(sizes), "max": max(sizes)}
    report["methods"]["graph"] = {t: score(run_graph(driver, q), t)
                                  for t, q in [("one_hop", one_hop), ("two_hop", two_hop)]}
    driver.close()
    for kind in ["product", "reaction"]:
        start = time.time()
        index = VectorIndex(records, kind)
        print(kind, "index built in", round(time.time() - start, 1), "s", flush=True)
        report["methods"][f"vector_{kind}"] = {t: score(run_vector(index, q), t)
                                               for t, q in [("one_hop", one_hop), ("two_hop", two_hop)]}
    report["reagents_above_degree_cap"] = sum(
        d > MAX_REAGENT_DEGREE for d in Counter(g for r in records for g in r["reagents"]).values())
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["methods"], indent=1))
    print(report["answer_set_size"])


if __name__ == "__main__":
    main()
