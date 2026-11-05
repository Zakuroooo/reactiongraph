"""Part 4: the MCP server. Four tools an agent can call over the reaction graph.

Each tool validates its input with RDKit first and returns {"error": ...} instead of raising, so a
bad SMILES gives the agent something it can read. Run: python -m src.mcp_server
"""
import os
from functools import lru_cache

from mcp.server.mcpserver import MCPServer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from src import graph
from src.vectors import VectorIndex

RDLogger.DisableLog("rdApp.*")

MAX_LIMIT = 50
mcp = MCPServer("reactiongraph")


@lru_cache(maxsize=1)
def _driver():
    return graph.connect()


@lru_cache(maxsize=1)
def _index():
    records = graph.read_records()
    return VectorIndex(records, "reaction")


def _canonical(smiles):
    """One canonical SMILES, or None. Every tool resolves its input through this first."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol) if mol is not None else None


def _clamp(limit):
    try:
        return max(1, min(MAX_LIMIT, int(limit)))
    except (TypeError, ValueError):
        return 10


@mcp.tool()
def molecule_info(smiles: str) -> dict:
    """Validate a SMILES string and return its canonical form, formula, weight and graph degree.

    Args:
        smiles: a molecule as SMILES, for example CC(=O)Oc1ccccc1C(=O)O for aspirin.
    """
    canonical = _canonical(smiles)
    if canonical is None:
        return {"error": "could not parse that SMILES", "input": smiles}
    mol = Chem.MolFromSmiles(canonical)
    rows = graph.query(_driver(), """
        MATCH (m:Molecule {smiles: $smiles})
        RETURN size([(m)<-[:HAS_PRODUCT]-() | 1]) AS made_by,
               size([(m)<-[:HAS_REACTANT]-() | 1]) AS used_as_reactant,
               size([(m)<-[:HAS_REAGENT]-() | 1]) AS used_as_reagent
    """, smiles=canonical)
    return {
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "in_graph": bool(rows),
        **(rows[0] if rows else {"made_by": 0, "used_as_reactant": 0, "used_as_reagent": 0}),
    }


@mcp.tool()
def find_routes_to(smiles: str, limit: int = 10) -> dict:
    """List reactions in the graph that make this molecule, with their reactants and patents.

    Args:
        smiles: the target molecule as SMILES.
        limit: how many reactions to return, 1 to 50.
    """
    canonical = _canonical(smiles)
    if canonical is None:
        return {"error": "could not parse that SMILES", "input": smiles}
    rows = graph.query(_driver(), graph.ROUTES_TO, smiles=canonical, limit=_clamp(limit))
    return {"query_molecule": canonical, "count": len(rows), "routes": rows}


@mcp.tool()
def reagents_for(smiles: str, limit: int = 10) -> dict:
    """List the reagents used in reactions that make this molecule, most used first.

    Args:
        smiles: the target molecule as SMILES.
        limit: how many reagents to return, 1 to 50.
    """
    canonical = _canonical(smiles)
    if canonical is None:
        return {"error": "could not parse that SMILES", "input": smiles}
    rows = graph.query(_driver(), graph.REAGENTS_FOR, smiles=canonical, limit=_clamp(limit))
    return {"query_molecule": canonical, "count": len(rows), "reagents": rows}


@mcp.tool()
def similar_reactions(smiles: str, limit: int = 5) -> dict:
    """Find reactions whose molecules look most like this one, by Morgan-fingerprint cosine similarity.

    Args:
        smiles: a molecule as SMILES. It does not have to be in the graph.
        limit: how many reactions to return, 1 to 50.
    """
    canonical = _canonical(smiles)
    if canonical is None:
        return {"error": "could not parse that SMILES", "input": smiles}
    hits = _index().search([canonical], k=_clamp(limit))
    return {"query_molecule": canonical, "count": len(hits), "reactions": [
        {"id": r["id"], "similarity": round(score, 4), "reactants": r["reactants"],
         "reagents": r["reagents"], "products": r["products"], "patents": sorted(r["patents"])}
        for r, score in hits]}


TOOLS = [molecule_info, find_routes_to, reagents_for, similar_reactions]

if __name__ == "__main__":
    if os.environ.get("REACTIONGRAPH_WARM") == "1":
        _index()  # pay the fingerprint cost before an agent's first call
    mcp.run()
