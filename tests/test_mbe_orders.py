from mbe_tools.mbe import MBEParams, generate_subsets_xyz
from mbe_tools.utils import Atom


def _frag(x):
    return [Atom("H", x, 0.0, 0.0)]


def test_generate_subsets_respects_orders():
    fragments = [_frag(0), _frag(1), _frag(2)]
    params = MBEParams(max_order=3, orders=[1, 3], backend="qchem")

    outs = list(generate_subsets_xyz(fragments, params))
    orders_seen = sorted({len(subset) for _, subset, _ in outs})
    assert orders_seen == [1, 3]
    # combinations: C(3,1)=3 and C(3,3)=1
    assert len(outs) == 4


def test_generate_subsets_default_max_order():
    fragments = [_frag(0), _frag(1)]
    params = MBEParams(max_order=2, backend="qchem")
    outs = list(generate_subsets_xyz(fragments, params))
    assert len(outs) == 3  # C(2,1)=2 plus C(2,2)=1
