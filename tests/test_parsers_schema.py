from mbe_tools.parsers.base import ParsedRecord
from mbe_tools.parsers.io import infer_metadata_from_path, parse_files


def test_parsed_record_to_json_contains_expected_keys():
    rec = ParsedRecord(
        job_id="job1",
        program="qchem",
        path="job1.out",
        status="ok",
        energy_hartree=-75.0,
        subset_size=1,
        subset_indices=[0],
        cp_correction=True,
    )
    data = rec.to_json()
    for key in [
        "job_id",
        "program",
        "program_detected",
        "status",
        "energy_hartree",
        "subset_size",
        "subset_indices",
        "cp_correction",
    ]:
        assert key in data
    assert data["job_id"] == "job1"
    assert data["subset_indices"] == [0]


def test_infer_metadata_from_f_token_path():
    meta = infer_metadata_from_path("/tmp/qchem_k2_f000-003_cp_deadbeef.out")
    assert meta["subset_size"] == 2
    assert meta["subset_indices"] == [0, 3]
    assert meta["cp_correction"] is True


def test_parse_files_sets_metadata_from_filename(tmp_path):
    path = tmp_path / "qchem_k2_f000-003_cp_deadbeef.out"
    path.write_text(
        "Q-Chem\nTotal energy in the final basis set = -10.0\n",
        encoding="utf-8",
    )
    recs = parse_files([str(path)], program="auto", infer_metadata=True)
    rec = recs[0]
    assert rec.program == "qchem"
    assert rec.subset_size == 2
    assert rec.subset_indices == [0, 3]
    assert rec.cp_correction is True
