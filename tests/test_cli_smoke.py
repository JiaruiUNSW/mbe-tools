import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mbe_tools.cli import app


runner = CliRunner()


def _write_water_xyz(tmp_path: Path) -> Path:
    text = """6
water
O 0.0 0.0 0.0
H 0.96 0.0 0.0
H -0.32 0.92 0.0
O 3.0 0.0 0.0
H 3.96 0.0 0.0
H 2.68 0.92 0.0
"""
    path = tmp_path / "water.xyz"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_fragment_water_heuristic(tmp_path):
    xyz = _write_water_xyz(tmp_path)
    out_xyz = tmp_path / "out.xyz"
    result = runner.invoke(app, ["fragment", str(xyz), "--out-xyz", str(out_xyz), "--n", "2", "--seed", "1"])
    assert result.exit_code == 0
    assert out_xyz.exists()


def test_cli_gen_subsets(tmp_path):
    xyz = _write_water_xyz(tmp_path)
    out_dir = tmp_path / "geoms"
    result = runner.invoke(app, ["gen", str(xyz), "--out-dir", str(out_dir), "--max-order", "2", "--backend", "qchem"])
    assert result.exit_code == 0
    files = list(out_dir.glob("*.geom"))
    assert len(files) == 3  # two singles + one pair for 2 fragments


def test_cli_parse_auto(tmp_path):
    q_path = tmp_path / "qchem_k1_f000_cp.out"
    q_path.write_text("Q-Chem 5.4\n Total energy in the final basis set = -5.0\n", encoding="utf-8")
    o_path = tmp_path / "orca_k1_f001_cp.out"
    o_path.write_text("O   R   C   A   5.0\n FINAL SINGLE POINT ENERGY   -10.0\n", encoding="utf-8")
    out_path = tmp_path / "parsed.jsonl"
    result = runner.invoke(app, [
        "parse",
        str(tmp_path),
        "--program",
        "auto",
        "--glob-pattern",
        "*.out",
        "--out",
        str(out_path),
    ])
    assert result.exit_code == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(ln) for ln in lines]
    programs = {r["program"] for r in recs}
    assert programs == {"qchem", "orca"}
    assert all(r.get("subset_size") == 1 for r in recs)


def test_cli_analyze_to_csv(tmp_path):
    pd = pytest.importorskip("pandas")
    jsonl = tmp_path / "parsed.jsonl"
    rows = [
        {"job_id": "a", "subset_size": 1, "energy_hartree": 0.1, "cpu_seconds": 1.0},
        {"job_id": "b", "subset_size": 2, "energy_hartree": 0.2, "cpu_seconds": 2.0},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    result = runner.invoke(app, ["analyze", str(jsonl), "--to-csv", str(out_csv)])
    assert result.exit_code == 0
    assert out_csv.exists()
