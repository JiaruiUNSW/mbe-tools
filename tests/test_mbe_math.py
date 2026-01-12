import pytest

from mbe_tools.mbe_math import assemble_mbe_energy


def test_assemble_mbe_energy_inclusion_exclusion():
    records = [
        {"subset_indices": [0], "energy_hartree": 1.0},
        {"subset_indices": [1], "energy_hartree": 2.0},
        {"subset_indices": [0, 1], "energy_hartree": 2.4},
    ]

    result = assemble_mbe_energy(records)
    contribs = result["contributions"]

    assert contribs[(0,)] == pytest.approx(1.0)
    assert contribs[(1,)] == pytest.approx(2.0)
    assert contribs[(0, 1)] == pytest.approx(-0.6)

    assert result["order_totals"][1] == pytest.approx(3.0)
    assert result["order_totals"][2] == pytest.approx(2.4)
    assert not result["missing_subsets"]


def test_missing_lower_order_subsets_reported():
    records = [
        {"subset_indices": [0, 1], "energy_hartree": -10.0},
    ]
    result = assemble_mbe_energy(records)
    assert (0,) in result["missing_subsets"]
    assert (1,) in result["missing_subsets"]
