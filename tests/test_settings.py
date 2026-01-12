import os
from pathlib import Path

import pytest

from mbe_tools.config import load_settings, Settings
from mbe_tools.hpc_templates import render_pbs_qchem, render_slurm_orca


def test_settings_defaults(tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        s = load_settings(env={}, cwd=str(tmp_path))
    assert isinstance(s, Settings)
    assert s.qchem_command is None
    assert s.orca_command is None


def test_settings_env_overrides():
    env = {"MBE_QCHEM_CMD": "qchem_custom", "MBE_ORCA_CMD": "orca_custom"}
    s = load_settings(env=env, cwd=os.getcwd())
    assert s.qchem_command == "qchem_custom"
    assert s.orca_command == "orca_custom"


def test_settings_project_overrides_user(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    global_cfg = home / ".config" / "mbe-tools" / "config.toml"
    local_cfg = project / "mbe.toml"
    global_cfg.parent.mkdir(parents=True)
    project.mkdir(parents=True)
    global_cfg.write_text("qchem_command = 'global_qchem'\n", encoding="utf-8")
    local_cfg.write_text("qchem_command = 'local_qchem'\n", encoding="utf-8")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(home))
        s = load_settings(env={}, cwd=str(project))
    assert s.qchem_command == "local_qchem"


def test_settings_propagation_into_templates():
    s = Settings(qchem_module="qcmod/1.0", orca_module="orcamod/2.0")
    pbs = render_pbs_qchem(job_name="job", module=s.qchem_module or "")
    slurm = render_slurm_orca(job_name="job", module=s.orca_module or "", command="orca_bin")
    assert "module load qcmod/1.0" in pbs
    assert "module load orcamod/2.0" in slurm
    assert "orca_bin" in slurm
