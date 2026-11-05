"""End-to-end check: start the real server over stdio and talk to it as an MCP client would.

The other tests call the tool functions directly. This one proves the server actually boots, that an
agent can list its tools and read their schemas, and that a call returns a result over the protocol.
Skipped when no Neo4j is reachable.
"""
import os
import sys

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from src import graph  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
EXPECTED_TOOLS = {"molecule_info", "find_routes_to", "reagents_for", "similar_reactions"}


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def needs_neo4j():
    try:
        driver = graph.connect()
        driver.verify_connectivity()
        driver.close()
    except Exception as exc:  # noqa: BLE001 - no database means skip, not fail
        pytest.skip(f"no Neo4j: {exc}")


async def _session(run):
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"],
                                   env={**os.environ, "PYTHONPATH": os.getcwd()})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await run(session)


@pytest.mark.anyio
async def test_server_lists_its_four_tools_with_schemas(needs_neo4j, anyio_backend):
    async def run(session):
        return await session.list_tools()

    tools = await _session(run)
    by_name = {t.name: t for t in tools.tools}
    assert EXPECTED_TOOLS <= set(by_name)
    for name in EXPECTED_TOOLS:
        tool = by_name[name]
        assert tool.description, f"{name} has no description for the agent to read"
        assert "smiles" in tool.input_schema["properties"], f"{name} must take a smiles argument"


@pytest.mark.anyio
async def test_a_real_tool_call_returns_the_canonical_molecule(needs_neo4j, anyio_backend):
    async def run(session):
        return await session.call_tool("molecule_info", {"smiles": "OC(=O)c1ccccc1OC(C)=O"})

    result = await _session(run)
    assert not result.is_error
    assert ASPIRIN in str(result.content)


@pytest.mark.anyio
async def test_a_bad_smiles_comes_back_as_a_readable_error_not_a_crash(needs_neo4j, anyio_backend):
    async def run(session):
        return await session.call_tool("molecule_info", {"smiles": "not_a_molecule"})

    result = await _session(run)
    assert "could not parse" in str(result.content)
