import os

from mbe_tools.config import load_settings, get_settings, use_settings, Settings


def test_settings_precedence(tmp_path, monkeypatch):
    # env (lowest), global, local, explicit (highest)
    env = {
        "MBE_QCHEM_CMD": "env_qchem",
        "MBE_ORCA_CMD": "env_orca",
    }

    # global config under mocked HOME
    home = tmp_path / "home"
    global_cfg = home / ".config" / "mbe-tools"
    global_cfg.mkdir(parents=True)
    (global_cfg / "config.toml").write_text("qchem_command = 'global_qchem'\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    # local project config
    project = tmp_path / "project"
    project.mkdir()
    (project / "mbe.toml").write_text("qchem_command = 'local_qchem'\n", encoding="utf-8")

    # explicit config file
    explicit = project / "explicit.toml"
    explicit.write_text("qchem_command = 'explicit_qchem'\n", encoding="utf-8")

    settings = load_settings(path=str(explicit), env=env, cwd=str(project))

    assert settings.qchem_command == "explicit_qchem"
    # orca_command falls back to env because not set elsewhere
    assert settings.orca_command == "env_orca"


def test_use_settings_context(monkeypatch):
    monkeypatch.delenv("MBE_QCHEM_CMD", raising=False)
    base = load_settings(env={})
    with use_settings(Settings(qchem_command="override")) as s:
        assert get_settings().qchem_command == "override"
        assert s.qchem_command == "override"
    # after context, cached base remains
    assert get_settings().qchem_command == base.qchem_command
