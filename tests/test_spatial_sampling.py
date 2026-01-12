from mbe_tools.cluster import (
    FragmentRecord,
    fragment_by_water_heuristic,
    spatial_sample_fragments,
    XYZ,
)
from mbe_tools.utils import Atom


def test_fragment_by_water_marks_ion_special():
    atoms = [
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.95, 0.0, 0.0),
        Atom("H", -0.95, 0.0, 0.0),
        Atom("Na", 5.0, 0.0, 0.0),
    ]
    xyz = XYZ(comment="", atoms=atoms)
    frags = fragment_by_water_heuristic(xyz, oh_cutoff=1.5)

    assert len(frags) == 2
    water = next(f for f in frags if f.label == "water")
    ion = next(f for f in frags if f.special)
    assert water.special is False
    assert ion.label == "ion"


def test_spatial_sampling_prefers_special_and_nearest():
    frags = [
        FragmentRecord([Atom("O", 0.0, 0.0, 0.0)], label="a", special=False),
        FragmentRecord([Atom("O", 1.0, 0.0, 0.0)], label="b", special=False),
        FragmentRecord([Atom("Na", 10.0, 0.0, 0.0)], label="ion", special=True),
    ]

    picked = spatial_sample_fragments(
        frags,
        n=2,
        seed=1,
        prefer_special=True,
        start="special",
    )
    labels = [f.label for f in picked]
    assert "ion" in labels
    assert "b" in labels  # closest to ion


def test_spatial_sampling_start_index_compact():
    frags = [
        FragmentRecord([Atom("O", 0.0, 0.0, 0.0)], label="a"),
        FragmentRecord([Atom("O", 1.0, 0.0, 0.0)], label="b"),
        FragmentRecord([Atom("O", 5.0, 0.0, 0.0)], label="c"),
        FragmentRecord([Atom("O", 10.0, 0.0, 0.0)], label="d"),
    ]

    picked = spatial_sample_fragments(
        frags,
        n=3,
        seed=7,
        prefer_special=False,
        start="index",
        start_index=0,
        k_neighbors=2,
    )
    labels = [f.label for f in picked]
    assert labels == ["a", "b", "c"]
