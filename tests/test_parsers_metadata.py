from mbe_tools.parsers.io import infer_metadata_from_path


def test_infer_metadata_from_path():
    meta = infer_metadata_from_path(
        "/tmp/qchem_cp_k3_f00-03-09_wb97m-v_def2-ma-QZVPP.out"
    )

    assert meta["subset_size"] == 3
    assert meta["subset_indices"] == [0, 3, 9]
    assert meta["cp_correction"] is True
    assert meta["method"].lower() == "wb97m-v"
    assert meta["basis"].lower().startswith("def2-ma")


def test_infer_metadata_new_job_id_pattern():
    meta = infer_metadata_from_path("/tmp/qchem_k2_1.3_abcd1234.out")
    assert meta["subset_size"] == 2
    # job_id stores 1-based indices; parser should convert to 0-based
    assert meta["subset_indices"] == [0, 2]
