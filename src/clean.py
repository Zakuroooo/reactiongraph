"""Part 1: clean a seeded sample of Lowe's USPTO reactions with RDKit.

Every molecule becomes one canonical SMILES (entity resolution), so two spellings of the
same molecule end up as one graph node. Run: python -m src.clean
"""
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RAW = Path("data/raw/x/1976_Sep2016_USPTOgrants_smiles.rsmi")
OUT = Path("data/clean/reactions.jsonl.gz")
COUNTS = Path("results/clean_counts.json")
SAMPLE_SIZE = 50_000
SEED = 42
ATOM_MAP = re.compile(r":\d+\]")


def canonical(smiles):
    """Return the canonical SMILES with atom-map numbers removed, or None if it does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _split(reaction):
    """Split 'a.b>c>d |f:0.1|' into three lists of parts, joining salts the |f:...| note groups."""
    body, _, note = reaction.strip().partition(" |")
    sides = body.split(">")
    if len(sides) != 3:
        return None
    parts = [(i, s) for i, side in enumerate(sides) for s in side.split(".") if s]
    groups = {}
    match = re.search(r"f:([\d.,]+)", note)
    if match:
        for g, group in enumerate(match.group(1).split(",")):
            for idx in filter(None, group.split(".")):
                groups[int(idx)] = g
    out = [[], [], []]
    seen = {}
    for idx, (side, smi) in enumerate(parts):
        g = groups.get(idx)
        if g is None:
            out[side].append(smi)
        elif g in seen:
            out[seen[g][0]][seen[g][1]] += "." + smi
        else:
            seen[g] = (side, len(out[side]))
            out[side].append(smi)
    return out


def parse_reaction(reaction):
    """Parse one reaction SMILES. Returns (record, None), or (None, reason) if it must be dropped.

    A 'reactant' with no mapped atoms gives no atoms to the product, so we treat it as a reagent.
    """
    sides = _split(reaction)
    if sides is None:
        return None, "bad_format"
    reactants, reagents, products = sides
    moved = [s for s in reactants if ":" not in s]
    reactants = [s for s in reactants if ":" in s]
    reagents = reagents + moved
    if not products:
        return None, "no_product"
    if not reactants:
        return None, "no_mapped_reactant"
    raw = {"reactants": reactants, "reagents": reagents, "products": products}
    clean = {}
    for role, items in raw.items():
        canon = [canonical(s) for s in items]
        if any(c is None for c in canon):
            return None, "unparseable_molecule"
        clean[role] = sorted(set(canon))
    # A product that also appears as a reactant is not something the reaction made.
    clean["products"] = [p for p in clean["products"] if p not in clean["reactants"]]
    if not clean["products"]:
        return None, "product_equals_reactant"
    clean["reagents"] = [r for r in clean["reagents"] if r not in clean["reactants"]]
    return {**clean, "moved_to_reagents": len(moved), "raw": raw}, None


def reaction_id(rec):
    key = ".".join(rec["reactants"]) + ">" + ".".join(rec["reagents"]) + ">" + ".".join(rec["products"])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def main():
    with RAW.open() as f:
        next(f)
        lines = f.readlines()
    random.seed(SEED)
    sample = random.sample(lines, SAMPLE_SIZE)
    drops = Counter()
    merged = {}
    mentions = 0
    moved = 0
    spellings = defaultdict(set)
    for line in sample:
        cols = line.rstrip("\n").split("\t")
        rec, why = parse_reaction(cols[0])
        if rec is None:
            drops[why] += 1
            continue
        for items in rec["raw"].values():
            for s in items:
                mentions += 1
                spellings[canonical(s)].add(ATOM_MAP.sub("]", s))
        moved += rec["moved_to_reagents"]
        rid = reaction_id(rec)
        if rid not in merged:
            merged[rid] = {"id": rid, "reactants": rec["reactants"], "reagents": rec["reagents"],
                           "products": rec["products"], "patents": {}}
        merged[rid]["patents"][cols[1]] = int(cols[3]) if cols[3].isdigit() else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt") as f:
        for rec in merged.values():
            f.write(json.dumps(rec) + "\n")
    parsed = SAMPLE_SIZE - sum(drops.values())
    all_spellings = sum(len(v) for v in spellings.values())
    counts = {
        "source_rows_total": len(lines),
        "sample_in": SAMPLE_SIZE,
        "seed": SEED,
        "dropped_total": sum(drops.values()),
        "dropped_by_reason": dict(drops),
        "parsed": parsed,
        "duplicate_reactions_merged": parsed - len(merged),
        "reactions_out": len(merged),
        "molecule_mentions": mentions,
        "distinct_spellings_atom_maps_removed": all_spellings,
        "unique_molecules_canonical": len(spellings),
        "spellings_merged": all_spellings - len(spellings),
        "molecules_with_2plus_spellings": sum(len(v) > 1 for v in spellings.values()),
        "unmapped_reactants_moved_to_reagents": moved,
        "patents": len({p for r in merged.values() for p in r["patents"]}),
    }
    COUNTS.write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
