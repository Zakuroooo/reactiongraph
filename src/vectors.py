"""Part 3: the vector baseline. Each reaction becomes a Morgan-fingerprint vector; search is cosine.

A Morgan fingerprint records the small atom neighbourhoods in a molecule, so similar molecules get
similar vectors. No model is needed. Two indexes for the eval, so the baseline gets a fair chance:
  product  - the vector of the reaction's product only
  reaction - reactants + reagents + product, each normalised then added, so reagents count too
A third index, inputs (reactants + reagents only), serves LoRA retrieval, which cannot see the product.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

BITS = 2048
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS)


def fingerprint(smiles_list):
    """Sparse 1 x BITS count vector for a list of SMILES, normalised to length 1."""
    total = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for bit, count in _generator.GetCountFingerprint(mol).GetNonzeroElements().items():
            total[bit] = total.get(bit, 0) + count
    if not total:
        return sparse.csr_matrix((1, BITS), dtype=np.float32)
    bits = list(total)
    values = np.array([total[b] for b in bits], dtype=np.float32)
    values /= np.linalg.norm(values)
    return sparse.csr_matrix((values, ([0] * len(bits), bits)), shape=(1, BITS), dtype=np.float32)


def _normalise(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1
    return sparse.diags(1 / norms, format="csr", dtype=np.float32) @ matrix


class VectorIndex:
    def __init__(self, records, kind="reaction"):
        self.records = records
        self.kind = kind
        cache = {}

        def fp(items):
            key = tuple(items)
            if key not in cache:
                cache[key] = fingerprint(items)
            return cache[key]

        if kind == "product":
            rows = [fp(r["products"]) for r in records]
        elif kind == "inputs":
            rows = [fp(r["reactants"]) + fp(r["reagents"]) for r in records]
        else:
            rows = [fp(r["reactants"]) + fp(r["reagents"]) + fp(r["products"]) for r in records]
        self.matrix = _normalise(sparse.vstack(rows, format="csr"))

    def search(self, smiles_list, k=20):
        """Return [(record, cosine score)] for the k most similar reactions, best first."""
        query = fingerprint(smiles_list)
        scores = (self.matrix @ query.T).toarray().ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.records[i], float(scores[i])) for i in top]


def ranked_answers(hits, field, exclude=(), k=50):
    """Walk the hits in order and collect unique molecules from `field` until we have k answers."""
    seen = []
    for record, _ in hits:
        for smiles in record[field]:
            if smiles not in seen and smiles not in exclude:
                seen.append(smiles)
                if len(seen) == k:
                    return seen
    return seen
