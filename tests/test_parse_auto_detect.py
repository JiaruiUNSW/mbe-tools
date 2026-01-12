from pytest import approx

from mbe_tools.parsers.io import detect_program, parse_files


def test_detect_program():
    q_text = "Q-Chem 5.4.1\n Total energy in the final basis set = -75.0"
    o_text = "O   R   C   A   5.0.3\n FINAL SINGLE POINT ENERGY   -222.0"

    assert detect_program(q_text) == "qchem"
    assert detect_program(o_text) == "orca"
    assert detect_program("no markers here") == "unknown"


def test_parse_files_auto(tmp_path):
    q_path = tmp_path / "q.out"
    q_path.write_text(
        "Q-Chem 5.4.1\n Total energy in the final basis set = -75.0\n Total job time: 3.0 s\n",
        encoding="utf-8",
    )
    o_path = tmp_path / "o.out"
    o_path.write_text(
        "O   R   C   A   5.0.3\n FINAL SINGLE POINT ENERGY   -222.0\n TOTAL RUN TIME: 0 days 0 hours 0 minutes 10.0 seconds\n",
        encoding="utf-8",
    )

    recs = parse_files([str(q_path), str(o_path)], program="auto", infer_metadata=False)

    assert recs[0].program == "qchem"
    assert recs[0].program_detected == "qchem"
    assert recs[0].energy_hartree == approx(-75.0)

    assert recs[1].program == "orca"
    assert recs[1].program_detected == "orca"
    assert recs[1].energy_hartree == approx(-222.0)
    assert recs[1].wall_seconds == approx(10.0)
