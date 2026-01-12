import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mbe_tools.cli import app


runner = CliRunner()


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_calc_units_and_monomer(tmp_path):
    jsonl = tmp_path / "parsed.jsonl"
    rows = [
        {
            "job_id": "a",
            "program": "qchem",
            "method": "m1",
            "basis": "b1",
            "grid": "g1",
            "cp_correction": True,
            "subset_size": 1,
            "subset_indices": [0],
            "energy_hartree": 1.0,
            "cpu_seconds": 1.0,
        },
        {
            "job_id": "b",
            "program": "qchem",
            "method": "m1",
            "basis": "b1",
            "grid": "g1",
            "cp_correction": True,
            "subset_size": 2,
            "subset_indices": [0, 1],
            "energy_hartree": 2.0,
            "cpu_seconds": 2.0,
        },
    ]
    _write_jsonl(jsonl, rows)

    result = runner.invoke(app, ["calc", str(jsonl), "--unit", "kcal", "--monomer", "0"])
    assert result.exit_code == 0
    assert "627.509474" in result.stdout  # 1 Hartree in kcal
    assert "kcal" in result.stdout
    assert "E(monomer 0)" in result.stdout


def test_calc_mixed_combo_rejected(tmp_path):
    jsonl = tmp_path / "parsed.jsonl"
    rows = [
        {"job_id": "a", "program": "qchem", "method": "m1", "basis": "b1", "grid": None, "cp_correction": True, "subset_size": 1, "subset_indices": [0], "energy_hartree": 0.0},
        {"job_id": "b", "program": "qchem", "method": "m2", "basis": "b1", "grid": None, "cp_correction": True, "subset_size": 1, "subset_indices": [1], "energy_hartree": 0.0},
    ]
    _write_jsonl(jsonl, rows)

    result = runner.invoke(app, ["calc", str(jsonl)])
    assert result.exit_code != 0
    assert "Mixed program" in result.stdout


def test_parse_embeds_geometry_from_singletons(tmp_path):
    out_path = tmp_path / "qchem_k1_f000_cp.out"
    out_text = """
Q-Chem 5.4.1
 Standard Nuclear Orientation (Angstroms)
 ---------------------------------------------------
   1 H   0.000000   0.000000   0.000000
   2 BQ  1.000000   0.000000   0.000000

 Total energy in the final basis set = -1.000000
"""
    out_path.write_text(out_text.strip() + "\n", encoding="utf-8")

    jsonl_out = tmp_path / "parsed.jsonl"
    result = runner.invoke(app, [
        "parse",
        str(tmp_path),
        "--program",
        "auto",
        "--glob-pattern",
        "*.out",
        "--out",
        str(jsonl_out),
    ])
    assert result.exit_code == 0

    lines = jsonl_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # cluster + calc
    cluster = json.loads(lines[0])
    calc = json.loads(lines[1])

    assert cluster.get("record_type") == "cluster"
    assert cluster.get("geometry_incomplete") is False
    monomers = cluster.get("monomers", [])
    assert len(monomers) == 1
    # ghost BQ should be dropped
    assert len(monomers[0].get("geometry_xyz", [])) == 1

    assert calc.get("program") == "qchem"
    assert calc.get("subset_size") == 1
    assert calc.get("energy_hartree") == -1.0