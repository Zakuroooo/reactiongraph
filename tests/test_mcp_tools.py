"""MCP tool tests against a small graph loaded into Neo4j for the test only.

Skipped when no Neo4j is reachable, so CI stays green without a database. Run them locally with:
docker start reactiongraph-neo4j && pytest
"""
import pytest

from src import graph

pytest.importorskip("neo4j")

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC = "O=C(O)c1ccccc1O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
PYRIDINE = "c1ccncc1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"

FIXTURE = [
    {"id": "test_rxn_1", "reactants": [SALICYLIC], "reagents": [ACETIC_ANHYDRIDE, PYRIDINE],
     "products": [ASPIRIN], "patents": {"USTEST0001": 1999}},
    {"id": "test_rxn_2", "reactants": ["Nc1ccc(O)cc1"], "reagents": [ACETIC_ANHYDRIDE],
     "products": [PARACETAMOL], "patents": {"USTEST0002": 2001}},
]


@pytest.fixture(scope="module")
def server():
    """Load the fixture reactions, hand the module its tools, then delete only what we added."""
    try:
        driver = graph.connect()
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - any connection problem means skip, not fail
        pytest.skip(f"no Neo4j: {exc}")
    graph.load(driver, FIXTURE)
    from src import mcp_server

    mcp_server._driver.cache_clear()
    mcp_server._driver = lambda: driver
    mcp_server._index = lambda: _FixtureIndex()
    yield mcp_server
    with driver.session() as session:
        session.run("MATCH (r:Reaction) WHERE r.id STARTS WITH 'test_rxn_' DETACH DELETE r").consume()
        session.run("MATCH (p:Patent) WHERE p.number STARTS WITH 'USTEST' DETACH DELETE p").consume()
        session.run("MATCH (m:Molecule) WHERE NOT (m)<-[]-() DETACH DELETE m").consume()
    driver.close()


class _FixtureIndex:
    """Stands in for the big fingerprint index so the tests stay fast."""

    def __init__(self):
        from src.vectors import VectorIndex

        self._index = VectorIndex(FIXTURE, "reaction")

    def search(self, smiles_list, k=5):
        return self._index.search(smiles_list, k=k)


def test_molecule_info_reads_a_known_molecule(server):
    out = server.molecule_info(" CC(=O)Oc1ccccc1C(=O)O ")
    assert out["canonical_smiles"] == ASPIRIN
    assert out["formula"] == "C9H8O4"
    assert out["molecular_weight"] == pytest.approx(180.16, abs=0.05)
    assert out["in_graph"] and out["made_by"] >= 1


def test_molecule_info_accepts_another_spelling_of_the_same_molecule(server):
    """Entity resolution in one line: a different spelling must land on the same node."""
    assert server.molecule_info("OC(=O)c1ccccc1OC(C)=O")["canonical_smiles"] == ASPIRIN


def test_molecule_info_reports_a_molecule_not_in_the_graph(server):
    """Uses an invented long chain; common molecules like ethanol really are in the loaded graph."""
    absent = "C" * 60
    out = server.molecule_info(absent)
    assert out["in_graph"] is False
    assert out["made_by"] == 0 and out["used_as_reactant"] == 0 and out["used_as_reagent"] == 0


def test_find_routes_to_returns_the_reaction_and_its_patent(server):
    routes = server.find_routes_to(ASPIRIN)["routes"]
    assert any(SALICYLIC in r["reactants"] and "USTEST0001" in r["patents"] for r in routes)


def test_reagents_for_ranks_shared_reagents_first(server):
    reagents = server.reagents_for(ASPIRIN)["reagents"]
    assert reagents[0]["reactions"] >= reagents[-1]["reactions"]
    assert ACETIC_ANHYDRIDE in [r["smiles"] for r in reagents]


def test_similar_reactions_puts_the_exact_molecule_first(server):
    hits = server.similar_reactions(ASPIRIN, limit=2)["reactions"]
    assert hits[0]["id"] == "test_rxn_1"
    assert hits[0]["similarity"] >= hits[-1]["similarity"]


@pytest.mark.parametrize("tool_name", ["molecule_info", "find_routes_to", "reagents_for", "similar_reactions"])
def test_every_tool_returns_an_error_instead_of_crashing(server, tool_name):
    for bad in ["", "   ", "not_a_molecule", "C(C"]:
        out = getattr(server, tool_name)(bad)
        assert "error" in out, f"{tool_name} did not report an error for {bad!r}"


def test_limit_is_clamped_not_trusted(server):
    assert server.find_routes_to(ASPIRIN, limit=10_000)["count"] <= server.MAX_LIMIT
    assert server.reagents_for(ASPIRIN, limit=0)["count"] >= 1
    assert server.reagents_for(ASPIRIN, limit="nonsense")["count"] >= 1


def test_all_four_tools_are_registered(server):
    assert len(server.TOOLS) == 4
    for tool in server.TOOLS:
        assert tool.__doc__, "each tool needs a docstring, which is its schema description"
