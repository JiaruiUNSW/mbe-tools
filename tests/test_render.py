from mbe_tools.input_builder import build_input_from_geom, render_orca_input, render_qchem_input


def test_render_qchem_includes_thresh_tole_and_scf():
    geom = "H 0 0 0\nH 0 0 1"
    text = render_qchem_input(
        geom,
        method="wb97m-v",
        basis="def2-ma-QZVPP",
        charge=0,
        multiplicity=1,
        thresh=1e-8,
        tole=12,
        scf_convergence="8",
        rem_extra="max_scf 200",
    )
    assert "thresh        1e-08" in text
    assert "tole          12" in text
    assert "scf_convergence 8" in text
    assert "max_scf 200" in text


def test_render_orca_includes_grid_and_scf():
    geom = "H 0 0 0\nH 0 0 1"
    text = render_orca_input(
        geom,
        method="wb97m-v",
        basis="def2-ma-QZVPP",
        charge=0,
        multiplicity=1,
        grid="GRID5",
        scf_convergence="TightSCF",
        keyword_line_extra="CPCM",
    )
    assert text.startswith("! wb97m-v def2-ma-QZVPP GRID5 TightSCF CPCM")
    assert "* xyz 0 1" in text


def test_build_input_from_geom_roundtrip(tmp_path):
    geom_path = tmp_path / "geom.txt"
    geom_path.write_text("H 0 0 0\nH 0 0 1\n", encoding="utf-8")
    qchem = build_input_from_geom(
        str(geom_path),
        backend="qchem",
        method="pbe0",
        basis="def2-svp",
        charge=1,
        multiplicity=2,
        thresh=1e-6,
        tole=8,
    )
    assert "$molecule" in qchem and "pbe0" in qchem and "def2-svp" in qchem

    orca = build_input_from_geom(
        str(geom_path),
        backend="orca",
        method="pbe0",
        basis="def2-svp",
        charge=0,
        multiplicity=1,
        grid="GRID3",
    )
    assert "! pbe0 def2-svp GRID3" in orca
    assert "* xyz 0 1" in orca
