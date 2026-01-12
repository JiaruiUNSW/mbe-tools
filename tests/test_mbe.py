import re

from mbe_tools.mbe import MBEParams, generate_subsets_xyz
from mbe_tools.utils import Atom


def _frags(count=3):
    return [[Atom("H", float(i), 0.0, 0.0)] for i in range(count)]


def test_generate_subsets_count_and_orders():
    frags = _frags(2)
    params = MBEParams(max_order=2, backend="qchem", cp_correction=True)
    subsets = list(generate_subsets_xyz(frags, params))
    assert len(subsets) == 3  # 2 singles + 1 pair
    sizes = [len(subset) for _, subset, _ in subsets]
    assert sizes.count(1) == 2
    assert sizes.count(2) == 1


def test_generate_subsets_orders_array():
    frags = _frags(3)
    params = MBEParams(max_order=3, backend="qchem", cp_correction=True)
    subsets = list(generate_subsets_xyz(frags, params, orders=[1, 3]))
    assert all(len(subset) in (1, 3) for _, subset, _ in subsets)
    assert any(len(subset) == 3 for _, subset, _ in subsets)
    assert not any(len(subset) == 2 for _, subset, _ in subsets)


def test_job_id_format_contains_hash():
    frags = _frags(2)
    params = MBEParams(max_order=1, backend="qchem", cp_correction=True)
    job_id, subset, geom = next(generate_subsets_xyz(frags, params))
    assert job_id.startswith("qchem_k1_")
    assert re.search(r"_[0-9a-f]{8}$", job_id)
    assert len(subset) == 1
    assert geom.strip()  # not empty


def test_cp_ghost_atoms_qchem():
    frags = _frags(2)
    params = MBEParams(max_order=1, backend="qchem", cp_correction=True)
    job_id, subset, geom = next(generate_subsets_xyz(frags, params))
    lines = geom.splitlines()
    assert any(line.strip().startswith("@H") for line in lines)  # ghost present
    assert any(line.strip().startswith("H") and not line.strip().startswith("@H") for line in lines)


def test_cp_ghost_atoms_orca():
    frags = _frags(2)
    params = MBEParams(max_order=1, backend="orca", cp_correction=True)
    job_id, subset, geom = next(generate_subsets_xyz(frags, params))
    lines = geom.splitlines()
    assert any(line.strip().startswith("H :") for line in lines)
    assert any(line.strip().startswith("H ") and not line.strip().startswith("H :") for line in lines)


def test_no_cp_no_ghosts():
    frags = _frags(2)
    params = MBEParams(max_order=1, backend="qchem", cp_correction=False)
    job_id, subset, geom = next(generate_subsets_xyz(frags, params))
    lines = geom.splitlines()
    assert all("@" not in line for line in lines)
