from mbe_tools.parsers.io import _parse_qchem_input_metadata, _parse_orca_input_metadata


def test_parse_qchem_input_metadata_extracts_method_and_basis():
    text = """
$molecule
0 1
H 0 0 0
H 0 0 0.9
$end

$rem
   method wb97m-v
   basis  def2-svp
   scf_convergence 8
$end
"""
    meta = _parse_qchem_input_metadata(text)
    assert meta["method"].lower() == "wb97m-v"
    assert meta["basis"].lower() == "def2-svp"


def test_parse_orca_input_metadata_header_tokens():
    text = """
! PBE0 def2-TZVPP GRID5 TightSCF
* xyz 0 1
O 0 0 0
H 0 0 0.95
H 0.8 0 0
*
"""
    meta = _parse_orca_input_metadata(text)
    assert meta["method"] == "PBE0"
    assert meta["basis"] == "def2-TZVPP"
    assert meta["grid"].upper() == "GRID5"
