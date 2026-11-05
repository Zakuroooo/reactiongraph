"""Cleaning and grading tests. These need no database, so CI runs them on every push."""
import pytest

from src.clean import canonical, parse_reaction
from src.grading import extract_smiles, grade, mcnemar_exact, wilson

ESTERIFICATION = ("[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]>S(=O)(=O)(O)O>"
                  "[CH3:1][C:2](=[O:3])[O:6][CH3:5]")


def test_canonical_merges_two_spellings_of_benzene():
    assert canonical("c1ccccc1") == canonical("C1=CC=CC=C1")


def test_canonical_strips_atom_map_numbers():
    assert canonical("[CH3:1][OH:2]") == canonical("CO")


def test_canonical_rejects_nonsense():
    assert canonical("not_a_molecule") is None
    assert canonical("C(C") is None


def test_parse_reaction_splits_the_three_roles():
    record, why = parse_reaction(ESTERIFICATION)
    assert why is None
    assert record["reactants"] == sorted({canonical("CC(=O)O"), canonical("CO")})
    assert record["reagents"] == [canonical("OS(=O)(=O)O")]
    assert record["products"] == [canonical("COC(C)=O")]


def test_unmapped_reactant_becomes_a_reagent():
    """A molecule on the left with no atom map gives no atoms to the product, so it is a reagent."""
    record, _ = parse_reaction("[CH3:1][OH:2].CCOCC>>[CH3:1][O:2][CH3:1]")
    assert canonical("CCOCC") in record["reagents"]
    assert canonical("CCOCC") not in record["reactants"]


def test_fragment_note_joins_a_salt_into_one_molecule():
    """|f:0.1| says parts 0 and 1 are one substance, so they must not become two nodes."""
    record, _ = parse_reaction("[Na+].[OH-].[CH3:1][OH:2]>>[CH3:1][O-:2] |f:0.1|")
    assert any("." in s for s in record["reagents"])


@pytest.mark.parametrize("reaction,reason", [
    ("CC(=O)O.CO", "bad_format"),
    ("[CH3:1][OH:2]>>", "no_product"),
    ("[CH3:1][OH:2]>>[CH3:1][OH:2]", "product_equals_reactant"),
    ("[CH3:1][Xx:2]>>[CH3:1][OH:2]", "unparseable_molecule"),
    ("CCO>>[CH3:1][OH:2]", "no_mapped_reactant"),
])
def test_bad_reactions_are_dropped_with_a_reason(reaction, reason):
    record, why = parse_reaction(reaction)
    assert record is None and why == reason


def test_grade_accepts_a_different_spelling_of_the_right_product():
    valid, exact = grade("c1ccccc1", "C1=CC=CC=C1")
    assert valid and exact


def test_grade_rejects_the_wrong_product():
    assert grade("CCO", "c1ccccc1") == (True, False)
    assert grade("", "c1ccccc1") == (False, False)
    assert grade("sorry", "c1ccccc1") == (False, False)


def test_a_prose_reply_can_still_count_as_valid_smiles():
    """Known limitation: 'I cannot answer' starts with I, which is iodine, so valid% is an upper bound.

    It never counts as correct, so exact match (the headline number) is unaffected.
    """
    valid, exact = grade("I cannot answer that", "c1ccccc1")
    assert valid is True and exact is False


def test_grade_is_strict_about_stereochemistry():
    assert grade("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O") == (True, False)


def test_extract_smiles_takes_the_first_token():
    assert extract_smiles("  `CCO` extra words ") == "CCO"
    assert extract_smiles("") == ""


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson(50, 100)
    assert low < 0.5 < high
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == pytest.approx(1.0)


def test_mcnemar_calls_a_clean_split_significant():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) > 0.5
    assert mcnemar_exact(20, 2) < 0.05
