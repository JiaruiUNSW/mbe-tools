from pytest import approx

from mbe_tools.parsers.qchem import parse_qchem_output
from mbe_tools.parsers.orca import parse_orca_output


def test_parse_qchem_sets_program_and_times():
    text = """
    Total energy in the final basis set = -75.123456
    Total job time: 12.5 s
    """
    rec = parse_qchem_output(text, path="job_qchem.out", job_id="job1")

    assert rec.program_detected == "qchem"
    assert rec.status == "ok"
    assert rec.energy_hartree == approx(-75.123456)
    assert rec.cpu_seconds == approx(12.5)
    assert rec.wall_seconds == approx(12.5)
    assert rec.error_reason is None


def test_parse_orca_sets_program_and_walltime():
    text = """
    FINAL SINGLE POINT ENERGY    -222.3344
    TOTAL RUN TIME: 0 days 1 hours 2 minutes 3.5 seconds
    """
    rec = parse_orca_output(text, path="job_orca.out", job_id="job2")

    assert rec.program_detected == "orca"
    assert rec.status == "ok"
    assert rec.energy_hartree == approx(-222.3344)
    assert rec.wall_seconds == approx(3600 + 120 + 3.5)
    assert rec.cpu_seconds == approx(rec.wall_seconds)
    assert rec.error_reason is None
