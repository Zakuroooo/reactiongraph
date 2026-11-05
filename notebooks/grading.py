"""The automatic grader for product prediction. RDKit decides; no human labels.

The notebook carries a copy of these functions. src/grade_lora.py re-grades the saved predictions
here with this copy, so the reported numbers do not depend on trusting the notebook's own summary.
"""
import math

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def extract_smiles(text):
    """Take the first token of the model's reply, which is where we told it to put the SMILES."""
    text = text.strip().strip("`").strip()
    if not text:
        return ""
    token = text.split()[0]
    return token.strip("`'\".,;:")


def canonical(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def grade(reply, product):
    """Return (valid, exact). Exact means the canonical SMILES match, stereochemistry included."""
    predicted = canonical(extract_smiles(reply))
    return predicted is not None, predicted == canonical(product)


def wilson(successes, n, z=1.96):
    """95% Wilson confidence interval for a proportion, as (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar p-value: did A and B differ on the questions where they disagree?"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
