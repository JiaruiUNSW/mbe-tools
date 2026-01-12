import math

import pytest

from mbe_tools.cluster import (
    FragmentRecord,
    fragment_by_connectivity,
    fragment_by_water_heuristic,
    read_xyz,
    sample_fragments,
    spatial_sample_fragments,
    write_xyz,
)
from mbe_tools.utils import Atom


def _write_xyz(tmp_path, atoms, comment=""):
    path = tmp_path / "sample.xyz"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for a in atoms:
            f.write(f"{a.element} {a.x} {a.y} {a.z}\n")
    return path


def test_read_write_roundtrip(tmp_path):
    atoms = [Atom("H", 0.0, 0.0, 0.0), Atom("O", 0.0, 0.0, 1.0)]
    path = _write_xyz(tmp_path, atoms, comment="hello")
    xyz = read_xyz(path)
    out = tmp_path / "out.xyz"
    write_xyz(out, [xyz.atoms], comment=xyz.comment)
    xyz2 = read_xyz(out)
    assert xyz2.comment == "hello"
    assert len(xyz2.atoms) == len(atoms)
    for a, b in zip(xyz.atoms, xyz2.atoms):
        assert a.element == b.element
        assert math.isclose(a.x, b.x, abs_tol=1e-8)
        assert math.isclose(a.y, b.y, abs_tol=1e-8)
        assert math.isclose(a.z, b.z, abs_tol=1e-8)


def test_read_xyz_invalid_atom_count(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("3\ncomment\nH 0 0 0\nH 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_xyz(path)


def _water_cluster_atoms(include_ion=False):
    atoms = [
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.32, 0.92, 0.0),
        Atom("O", 3.0, 0.0, 0.0),
        Atom("H", 3.96, 0.0, 0.0),
        Atom("H", 2.68, 0.92, 0.0),
    ]
    if include_ion:
        atoms.append(Atom("Na", 6.0, 0.0, 0.0))
    return atoms


def test_fragment_water_heuristic_basic(tmp_path):
    path = _write_xyz(tmp_path, _water_cluster_atoms(), comment="water")
    xyz = read_xyz(path)
    frags = fragment_by_water_heuristic(xyz)
    assert len(frags) == 2
    for frag in frags:
        assert len(frag.atoms) == 3
        assert frag.label == "water"


def test_fragment_water_heuristic_ionic(tmp_path):
    path = _write_xyz(tmp_path, _water_cluster_atoms(include_ion=True), comment="ion")
    xyz = read_xyz(path)
    frags = fragment_by_water_heuristic(xyz)
    labels = [f.label for f in frags]
    specials = [f.special for f in frags]
    assert "ion" in labels
    assert any(specials)
    water_frags = [f for f in frags if f.label == "water"]
    assert all(len(f.atoms) == 3 for f in water_frags)


def _methanol_molecule(origin_x=0.0):
    return [
        Atom("C", origin_x, 0.0, 0.0),
        Atom("O", origin_x + 1.4, 0.0, 0.0),
        Atom("H", origin_x - 0.6, 0.9, 0.0),
        Atom("H", origin_x - 0.6, -0.9, 0.0),
        Atom("H", origin_x, 0.0, 1.0),
        Atom("H", origin_x + 1.8, 0.6, 0.0),
    ]


def _ethanol_molecule(origin_x=0.0):
    return [
        Atom("C", origin_x, 0.0, 0.0),
        Atom("C", origin_x + 1.5, 0.0, 0.0),
        Atom("O", origin_x + 2.9, 0.0, 0.0),
        Atom("H", origin_x - 0.6, 0.9, 0.0),
        Atom("H", origin_x - 0.6, -0.9, 0.0),
        Atom("H", origin_x + 1.5, 1.0, 0.0),
        Atom("H", origin_x + 1.5, -1.0, 0.0),
        Atom("H", origin_x + 2.9, 0.8, 0.0),
        Atom("H", origin_x + 2.9, -0.8, 0.0),
    ]


def _benzene_molecule(shift_z=0.0):
    atoms = []
    import math as _m

    for i in range(6):
        angle = 2 * _m.pi * i / 6
        x = 1.4 * _m.cos(angle)
        y = 1.4 * _m.sin(angle)
        atoms.append(Atom("C", x, y, shift_z))
        hx = 2.4 * _m.cos(angle)
        hy = 2.4 * _m.sin(angle)
        atoms.append(Atom("H", hx, hy, shift_z))
    return atoms


def test_fragment_connectivity_molecules(tmp_path):
    atoms = _methanol_molecule() + _ethanol_molecule(origin_x=6.0) + _benzene_molecule(shift_z=4.0)
    path = _write_xyz(tmp_path, atoms, comment="molecules")
    xyz = read_xyz(path)
    frags = fragment_by_connectivity(xyz)
    labels = [f.label for f in frags]
    assert "methanol" in labels
    assert "ethanol" in labels
    assert labels.count("benzene") == 1
    sizes = sorted(len(f.atoms) for f in frags)
    assert 6 in sizes and 9 in sizes and 12 in sizes


def test_fragment_connectivity_benzene_dimer(tmp_path):
    atoms = _benzene_molecule(shift_z=0.0) + _benzene_molecule(shift_z=5.0)
    path = _write_xyz(tmp_path, atoms, comment="benzene2")
    xyz = read_xyz(path)
    frags = fragment_by_connectivity(xyz)
    assert len(frags) == 2
    assert all(f.label == "benzene" for f in frags)


def test_sample_random_reproducible(tmp_path):
    xyz = read_xyz(_write_xyz(tmp_path, _water_cluster_atoms(include_ion=True)))
    frags = fragment_by_water_heuristic(xyz)
    pick1 = sample_fragments(frags, n=2, seed=123)
    pick2 = sample_fragments(frags, n=2, seed=123)
    assert [p.centroid for p in pick1] == [p.centroid for p in pick2]


def test_sample_random_require_ion(tmp_path):
    xyz = read_xyz(_write_xyz(tmp_path, _water_cluster_atoms(include_ion=True)))
    frags = fragment_by_water_heuristic(xyz)
    picks = sample_fragments(frags, n=2, seed=7, require_ion=True)
    assert any(getattr(p, "special", False) for p in picks)


def test_spatial_sampling_reproducible():
    frags = [
        FragmentRecord([Atom("O", 0, 0, 0)], label="ion", special=True),
        FragmentRecord([Atom("H", 5, 0, 0)], label="water", special=False),
        FragmentRecord([Atom("H", 0, 5, 0)], label="water", special=False),
    ]
    pick1 = spatial_sample_fragments(frags, n=2, seed=11, prefer_special=True)
    pick2 = spatial_sample_fragments(frags, n=2, seed=11, prefer_special=True)
    assert [p.centroid for p in pick1] == [p.centroid for p in pick2]
    assert any(p.special for p in pick1)


def test_spatial_sampling_prefer_special_included():
    frags = [
        FragmentRecord([Atom("H", 5, 0, 0)], label="water", special=False),
        FragmentRecord([Atom("H", 0, 5, 0)], label="water", special=False),
        FragmentRecord([Atom("Na", 0, 0, 0)], label="ion", special=True),
    ]
    picked = spatial_sample_fragments(frags, n=2, seed=3, prefer_special=True)
    assert any(p.special for p in picked)
