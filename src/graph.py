"""Part 2: the Neo4j knowledge graph. Loading plus the Cypher queries the eval and MCP server share.

Run: python -m src.graph   (loads data/clean/reactions.jsonl.gz, saves results/graph_counts.json)
"""
import gzip
import json
import os
from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase

CLEAN = Path("data/clean/reactions.jsonl.gz")
COUNTS = Path("results/graph_counts.json")
BATCH = 2000

CONSTRAINTS = [
    "CREATE CONSTRAINT molecule_smiles IF NOT EXISTS FOR (m:Molecule) REQUIRE m.smiles IS UNIQUE",
    "CREATE CONSTRAINT reaction_id IF NOT EXISTS FOR (r:Reaction) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT patent_number IF NOT EXISTS FOR (p:Patent) REQUIRE p.number IS UNIQUE",
]

LOAD = """
UNWIND $rows AS row
MERGE (r:Reaction {id: row.id})
FOREACH (s IN row.reactants | MERGE (m:Molecule {smiles: s}) MERGE (r)-[:HAS_REACTANT]->(m))
FOREACH (s IN row.reagents  | MERGE (m:Molecule {smiles: s}) MERGE (r)-[:HAS_REAGENT]->(m))
FOREACH (s IN row.products  | MERGE (m:Molecule {smiles: s}) MERGE (r)-[:HAS_PRODUCT]->(m))
FOREACH (p IN row.patents   | MERGE (x:Patent {number: p.number}) SET x.year = p.year MERGE (r)-[:REPORTED_IN]->(x))
"""

SET_DEGREE = """
UNWIND $rows AS row
MATCH (m:Molecule {smiles: row.smiles}) SET m.reagent_degree = row.degree
"""

# One hop: which reagents were used in reactions that make this molecule? Most used first.
REAGENTS_FOR = """
MATCH (:Molecule {smiles: $smiles})<-[:HAS_PRODUCT]-(r:Reaction)-[:HAS_REAGENT]->(g:Molecule)
RETURN g.smiles AS smiles, count(DISTINCT r) AS reactions
ORDER BY reactions DESC, smiles
LIMIT $limit
"""

# Two hops: molecule -> its reaction -> a rare reagent -> another reaction -> that reaction's product.
OTHER_PRODUCTS = """
MATCH (x:Molecule {smiles: $smiles})<-[:HAS_PRODUCT]-(:Reaction)-[:HAS_REAGENT]->(g:Molecule)
WHERE g.reagent_degree <= $max_degree
MATCH (g)<-[:HAS_REAGENT]-(other:Reaction)-[:HAS_PRODUCT]->(p:Molecule)
WHERE p <> x
RETURN p.smiles AS smiles, count(DISTINCT g) AS shared
ORDER BY shared DESC, smiles
LIMIT $limit
"""

ROUTES_TO = """
MATCH (:Molecule {smiles: $smiles})<-[:HAS_PRODUCT]-(r:Reaction)
OPTIONAL MATCH (r)-[:REPORTED_IN]->(p:Patent)
WITH r, collect(DISTINCT p.number) AS patents
RETURN r.id AS id,
       [(r)-[:HAS_REACTANT]->(m) | m.smiles] AS reactants,
       [(r)-[:HAS_REAGENT]->(m) | m.smiles] AS reagents,
       patents
ORDER BY id
LIMIT $limit
"""


def connect():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "reactiongraph")
    return GraphDatabase.driver(uri, auth=(user, password))


def load(driver, records):
    """Load reaction records into the graph. MERGE means running it twice changes nothing."""
    rows = [{**r, "patents": [{"number": n, "year": y} for n, y in r["patents"].items()]} for r in records]
    degree = Counter(s for r in records for s in r["reagents"])
    with driver.session() as session:
        for statement in CONSTRAINTS:
            session.run(statement).consume()
        for i in range(0, len(rows), BATCH):
            session.run(LOAD, rows=rows[i:i + BATCH]).consume()
        degrees = [{"smiles": s, "degree": d} for s, d in degree.items()]
        for i in range(0, len(degrees), BATCH):
            session.run(SET_DEGREE, rows=degrees[i:i + BATCH]).consume()


def query(driver, cypher, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(cypher, **params)]


def counts(driver):
    out = {}
    for label in ["Reaction", "Molecule", "Patent"]:
        out[label] = query(driver, f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
    for rel in ["HAS_REACTANT", "HAS_REAGENT", "HAS_PRODUCT", "REPORTED_IN"]:
        out[rel] = query(driver, f"MATCH ()-[e:{rel}]->() RETURN count(e) AS c")[0]["c"]
    return out


def read_records(path=CLEAN):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def main():
    driver = connect()
    load(driver, read_records())
    result = counts(driver)
    COUNTS.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    driver.close()


if __name__ == "__main__":
    main()
