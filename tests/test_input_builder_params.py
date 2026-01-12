from mbe_tools.input_builder import render_qchem_input, render_orca_input


def test_render_qchem_includes_thresh_tole():
    geom = "H 0 0 0"
    text = render_qchem_input(
        geom,
        method="wb97m-v",
        basis="def2",
        charge=0,
        multiplicity=1,
        thresh=14,
        tole=8,
        scf_convergence="tight",
    )
    assert "thresh        14" in text
    assert "tole          8" in text
    assert "scf_convergence tight" in text


def test_render_orca_includes_grid_and_scf():
    geom = "H 0 0 0"
    text = render_orca_input(
        geom,
        method="wb97m-v",
        basis="def2",
        charge=0,
        multiplicity=1,
        grid="GRID5",
        scf_convergence="TightSCF",
        keyword_line_extra="D3BJ",
    )
    header = text.splitlines()[0]
    assert "GRID5" in header
    assert "TightSCF" in header
    assert "D3BJ" in header
