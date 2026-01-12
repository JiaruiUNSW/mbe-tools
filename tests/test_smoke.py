def test_import():
    import mbe_tools
    assert hasattr(mbe_tools, "read_xyz")
    assert hasattr(mbe_tools, "__version__")
    assert isinstance(mbe_tools.__version__, str)
