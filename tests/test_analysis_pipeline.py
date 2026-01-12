import pytest

pd = pytest.importorskip("pandas")
from mbe_tools import analysis


def test_read_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n{}\n\n{\"a\":1}\n", encoding="utf-8")
    rows = analysis.read_jsonl(path)
    assert len(rows) == 2
    assert rows[1]["a"] == 1


def test_summarize_by_order_groups_counts():
    df = pd.DataFrame(
        {
            "job_id": ["a", "b", "c"],
            "subset_size": [1, 1, 2],
            "energy_hartree": [-1.0, -1.1, -2.0],
            "cpu_seconds": [1.0, 2.0, 3.0],
        }
    )
    summary = analysis.summarize_by_order(df)
    assert set(summary["subset_size"]) == {1, 2}
    row1 = summary[summary["subset_size"] == 1].iloc[0]
    assert row1["n"] == 2
    assert pytest.approx(row1["cpu_total"], rel=1e-6) == 3.0


def test_compute_delta_energy_reference_order():
    df = pd.DataFrame(
        {
            "subset_size": [1, 2, 2],
            "energy_hartree": [0.5, 0.7, 0.9],
        }
    )
    out = analysis.compute_delta_energy(df, reference_order=1)
    ref_val = out[out["subset_size"] == 1]["delta_energy_hartree_vs_ref"].iloc[0]
    assert pytest.approx(ref_val, abs=1e-12) == 0.0
    deltas = out[out["subset_size"] == 2]["delta_energy_hartree_vs_ref"].tolist()
    assert all(d > 0 for d in deltas)


def test_strict_mbe_orders_returns_rows():
    records = [
        {"subset_indices": [0], "energy_hartree": -1.0},
        {"subset_indices": [1], "energy_hartree": -1.1},
        {"subset_indices": [0, 1], "energy_hartree": -2.2},
    ]
    rows, missing = analysis.strict_mbe_orders(records, max_order=2)
    assert rows
    assert not missing
