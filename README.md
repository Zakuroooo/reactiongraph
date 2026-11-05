# ReactionGraph

A reaction knowledge graph built from US patent reactions, with MCP tools for agents, a measured
comparison of graph retrieval against vector search, and a LoRA fine-tune graded by RDKit.

Built in one day (5 November 2026) as a learning project. **No users, no chemist has reviewed it, and
every number below comes from a file in `results/`.** Where a result is weak or the test is biased,
this README says so.

---

## What it does

| Part | What | Where |
|---|---|---|
| 1 | Cleans patent reactions, merging molecule spellings with RDKit canonical SMILES | `src/clean.py` |
| 2 | Loads them into a Neo4j graph | `src/graph.py` |
| 3 | Scores graph retrieval against two vector baselines | `src/retrieval_eval.py` |
| 4 | Serves four tools to an agent over MCP | `src/mcp_server.py` |
| 5 | LoRA fine-tune vs base, few-shot and retrieval prompting | `notebooks/run_lora.py` |

## Data

Daniel Lowe's *Chemical reactions from US patents (1976–Sep2016)*,
[figshare 5104873](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873).
**Licence checked on the record itself before downloading: CC0** (`license.name: "CC0"` in
`data/raw/figshare_record.json`), so it is free to use. The archive's MD5 matches the one figshare
publishes.

A seeded sample of **50,000** of the 1,808,937 reactions in the grants file. The data is text-mined
from patents, so it contains mistakes that no amount of cleaning here removes.

### Cleaning (`results/clean_counts.json`)

| | |
|---|---|
| Reactions sampled (seed 42) | 50,000 |
| Dropped | 30 (29 unparseable, 1 product identical to a reactant) |
| Duplicate reactions merged | 1,446 |
| **Reactions kept** | **48,524** |
| Molecule mentions | 254,898 |
| Distinct spellings (atom maps removed) | 112,184 |
| **Unique molecules after canonicalisation** | **100,444** |
| Spellings merged into an existing molecule | 11,740 |
| Molecules that had 2+ spellings | 6,720 |

That last row is the entity resolution: 6,720 molecules were written more than one way in the
patents, and canonical SMILES collapses each group to one node.

Two further decisions, both visible in `src/clean.py`:
- A left-hand molecule with **no atom-map numbers** contributes no atoms to the product, so it is
  treated as a reagent, not a reactant (51,549 of them).
- A `|f:0.1|` note means two fragments are one substance (a salt), so they are kept as one molecule
  rather than becoming two nodes.

## The graph (`results/graph_counts.json`)

```
(Reaction)-[:HAS_REACTANT]->(Molecule)
(Reaction)-[:HAS_REAGENT]->(Molecule)
(Reaction)-[:HAS_PRODUCT]->(Molecule)
(Reaction)-[:REPORTED_IN]->(Patent)
```

48,524 Reaction nodes · 100,444 Molecule nodes · 26,609 Patent nodes · 297,225 relationships.
Uniqueness constraints on `Molecule.smiles`, `Reaction.id` and `Patent.number`; the loader uses
`MERGE`, so re-running it changes nothing.

```bash
docker run -d --name reactiongraph-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/reactiongraph neo4j:5
python -m src.clean && python -m src.graph
```

## Part 3: graph retrieval vs vector search

200 one-hop and 200 two-hop questions, generated from the cleaned data in plain Python
(`src/retrieval_eval.py`), scored by recall@k with a bootstrap 95% CI.

- **One-hop:** which reagents are used in reactions that make molecule X?
- **Two-hop:** which other products come from reactions sharing a rare reagent with X?
  (A reagent used in more than 50 reactions is a solvent or base, not a useful link, so those are
  excluded — 167 reagents are above that cap.)

Vector baselines use Morgan count fingerprints (radius 2, 2048 bits) with cosine similarity, reading
answers from the 200 nearest reactions. Two indexes, so the baseline gets a fair shot: one over the
product only, one over reactants + reagents + product.

### Results (`results/retrieval.json`), recall@10

| Method | One-hop | Two-hop | Median latency |
|---|---|---|---|
| **Graph (Cypher)** | **1.000** [1.000, 1.000] | **0.613** [0.569, 0.658] | 1.3 ms |
| Vector, product index | **1.000** [1.000, 1.000] | 0.049 [0.027, 0.074] | 9.8 ms |
| Vector, reaction index | 0.934 [0.901, 0.964] | 0.025 [0.011, 0.044] | 12.8 ms |

### 🔴 Read this before quoting those numbers

**The test favours the graph.** The correct answers were computed by following exactly the links the
graph stores. A fair test would have questions and answers from an outside source — a chemist, or a
separate dataset. This measures "can each method recover the structure we know is there", not "which
helps a chemist more".

**And the honest reading of the one-hop row:** vector search on a product index **ties the graph
exactly** (1.000 vs 1.000). It is not a near miss — it is the same number, because the nearest
neighbour of molecule X is usually the very reaction that makes X, so the baseline reads the right
reagents straight off it. A single hop over an exact-match key is not a problem a graph is needed for.

The graph's real advantage is **two-hop**: 0.613 against 0.049, a 12× gap, because following a shared
rare reagent to a different reaction is not a similarity question at all. The two molecules at either
end of that path often look nothing alike, so no embedding brings them together.

Two-hop answer sets are large (median 18 answers, max 78), which is why even the graph only reaches
0.613 at k=10 — it cannot fit 18 answers into 10 slots. At k=50 the graph reaches 0.991.

## Part 4: the MCP server

Four tools, each with a docstring that becomes its schema description (`src/mcp_server.py`):

| Tool | Returns |
|---|---|
| `molecule_info` | canonical SMILES, formula, weight, rings, and how the molecule is used in the graph |
| `find_routes_to` | reactions that make a molecule, with reactants, reagents and patent numbers |
| `reagents_for` | reagents used to make a molecule, most-used first |
| `similar_reactions` | fingerprint-similar reactions (the Part 3 vector search) |

Every tool resolves its input through RDKit first and returns `{"error": ...}` for anything it cannot
parse, so a bad SMILES gives an agent something readable instead of a stack trace. `limit` is clamped
to 1–50 rather than trusted.

```bash
python -m src.mcp_server
```

## Tests

**33 tests, `pytest -q`.** They cover the cleaning decisions (atom-map handling, salt fragments, each
drop reason), the grader (different spellings of the right product count as correct, stereochemistry
does not), the statistics, and all four MCP tools against a small graph loaded for the test only.

Three of them are end-to-end: they start the server over stdio and talk to it as an agent would —
list the tools, read their schemas, call one, and check that a bad SMILES comes back as a readable
error. Every tool is also checked against four kinds of bad input, and `limit` is clamped, not trusted.

GitHub Actions runs lint and the whole suite on every push, with **Neo4j as a service container**, so
the graph tests really execute there rather than skipping. Run locally they skip themselves when no
database is reachable.

**One honest limitation the tests document:** the grader takes the model's first token, and a refusal
starting with "I" parses as iodine. So "valid SMILES %" is an upper bound. Exact match — the number
that matters — is unaffected.

## Part 5: LoRA fine-tune (run on a Kaggle T4)

Task: given reactants and reagents, predict the product. RDKit canonicalises both sides and compares,
so the grader needs no human labels.

Split by **whole patent** (`results/lora_data_stats.json`), so a near-copy of a test reaction cannot
sit in the training set:

| | |
|---|---|
| Train / val / test | 20,000 / 200 / 500 |
| Patents in both train and test | **0** |
| Test products that also appear in training | 2 of 500 (0.4%) |
| Copying the nearest training reaction's product | **0.0%** correct [0.0, 0.8] |

That last row is the baseline worth knowing: just copying the most similar training reaction's
product gets **none** of the 500 right, so the task is not solvable by lookup.

Four methods, same 500 questions: **base** (cold), **few-shot** (3 fixed examples), **retrieval**
(the 3 most similar training reactions), **LoRA** (frozen model, trained adapters, asked cold).
Scored on valid SMILES %, exact match %, each with a Wilson 95% CI, plus seconds per answer and an
exact McNemar test on each pair.

Steps to run it: **`notebooks/KAGGLE-STEPS.md`**. The predictions are re-graded locally by
`src/grade_lora.py`, which reads only the raw replies — so the reported number does not depend on
trusting the notebook's own summary.

### Results

Not run yet. This section will hold the measured numbers, or state plainly that the run did not
finish. **Expect weak numbers:** a 1.5B general model is not a chemistry model, and LoRA may not beat
retrieval. That is still a real result.

## Limits

- One day, a 50,000-reaction sample, built with AI assistance.
- Patent text-mining has errors; nothing here fixes them.
- No chemist has checked any output.
- The retrieval comparison is biased toward the graph (see above) and the one-hop half shows no graph
  advantage at all.
- Canonical SMILES treats a salt, a free base and a different tautomer as different molecules, so
  entity resolution here is narrower than a chemist's idea of "the same compound".
- One step only: this does not plan or rank multi-step routes.
- Not a copy of any product; it is a small learning exercise on public data.

## Layout

```
src/clean.py           sampling, RDKit canonicalisation, drop/merge counts
src/graph.py           Neo4j schema, loader, the shared Cypher queries
src/vectors.py         Morgan-fingerprint index and cosine search
src/retrieval_eval.py  question generation, recall@k, bootstrap CIs
src/mcp_server.py      the four MCP tools
src/grading.py         the RDKit grader, Wilson CI, exact McNemar
src/grade_lora.py      re-grades the Kaggle predictions locally
src/check_results.py   fails if a number in this README is not in results/
tests/                 33 tests: cleaning, grading, statistics, MCP tools, MCP over stdio
notebooks/run_lora.py  the GPU experiment
results/               every number quoted anywhere comes from these files
```
