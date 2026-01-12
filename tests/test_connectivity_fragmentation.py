from mbe_tools.cluster import fragment_by_connectivity, XYZ
from mbe_tools.utils import Atom


def test_connectivity_water_and_ion():
    atoms = [
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", -0.24, 0.93, 0.0),
        Atom("Na", 5.0, 0.0, 0.0),
    ]
    xyz = XYZ(comment="", atoms=atoms)
    frags = fragment_by_connectivity(xyz)
    labels = sorted([f.label for f in frags])
    specials = [f.special for f in frags if f.label == "ion"]
    assert labels == ["ion", "water"]
    assert specials == [True]


def test_connectivity_methanol_and_ethanol():
    # Simplified geometries just to form connectivity; radii-based bonding should connect each molecule separately
    methanol_atoms = [
        Atom("C", 0, 0, 0), Atom("O", 1.4, 0, 0), Atom("H", -0.5, 0.9, 0), Atom("H", -0.5, -0.9, 0),
        Atom("H", 0, 0, 1.0), Atom("H", 1.9, 0.9, 0)
    ]
    ethanol_atoms = [
        Atom("C", 5, 0, 0), Atom("C", 6.4, 0, 0), Atom("O", 7.8, 0, 0),
        Atom("H", 4.5, 0.9, 0), Atom("H", 4.5, -0.9, 0), Atom("H", 5, 0, 1.0),
        Atom("H", 6.4, 0.9, 0), Atom("H", 6.4, -0.9, 0), Atom("H", 6.4, 0, 1.0)
    ]
    atoms = methanol_atoms + ethanol_atoms
    xyz = XYZ(comment="", atoms=atoms)
    frags = fragment_by_connectivity(xyz)

    labels = sorted([f.label for f in frags])
    assert labels == ["ethanol", "methanol"]
    assert all(not f.special for f in frags)


def test_connectivity_benzene():
    # Rough hexagon to connect
    atoms = [
        Atom("C", 0, 1.4, 0), Atom("C", 1.2, 0.7, 0), Atom("C", 1.2, -0.7, 0),
        Atom("C", 0, -1.4, 0), Atom("C", -1.2, -0.7, 0), Atom("C", -1.2, 0.7, 0),
        Atom("H", 0, 2.4, 0), Atom("H", 2.1, 1.2, 0), Atom("H", 2.1, -1.2, 0),
        Atom("H", 0, -2.4, 0), Atom("H", -2.1, -1.2, 0), Atom("H", -2.1, 1.2, 0),
    ]
    xyz = XYZ(comment="", atoms=atoms)
    frags = fragment_by_connectivity(xyz)
    assert len(frags) == 1
    assert frags[0].label == "benzene"
    assert frags[0].special is False
