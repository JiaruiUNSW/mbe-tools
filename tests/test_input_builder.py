from mbe_tools.input_builder import render_qchem_input, render_orca_input


def test_render_qchem_input_includes_method_basis_and_geom():
    geom = "H 0 0 0\nH 0 0 0.9"
    text = render_qchem_input(geom, method="wb97m-v", basis="def2-svp", charge=0, multiplicity=1)
    assert "$molecule" in text
    assert "method        wb97m-v" in text
    assert "basis         def2-svp" in text
    assert "H 0 0 0" in text


def test_render_orca_input_header_and_geom():
    geom = "O 0 0 0\nH 0 0 0.95\nH 0.8 0 0"
    text = render_orca_input(geom, method="wB97M-V", basis="def2-TZVPD", charge=-1, multiplicity=1, keyword_line_extra="TightSCF")
    assert text.startswith("! wB97M-V def2-TZVPD TightSCF")
    assert "* xyz -1 1" in text
    assert "O 0 0 0" in text
