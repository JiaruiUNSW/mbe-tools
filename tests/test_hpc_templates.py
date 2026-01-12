from mbe_tools.hpc_templates import render_pbs_qchem, render_slurm_orca


def test_render_pbs_qchem_includes_resources():
    txt = render_pbs_qchem(job_name="job1", walltime="10:00:00", ncpus=8, mem_gb=16, queue="normal", project="proj")
    assert "#PBS -N job1" in txt
    assert "#PBS -l walltime=10:00:00,mem=16000Mb,ncpus=8" in txt
    assert "#PBS -q normal" in txt
    assert "#PBS -P proj" in txt
    assert "qchem -np 8" in txt


def test_render_slurm_orca_includes_resources():
    txt = render_slurm_orca(job_name="job2", walltime="02:30:00", ntasks=2, cpus_per_task=4, mem_gb=12.5, partition="work", account="acc", qos="low", module="orca/6", command="orca6")
    assert "#SBATCH --job-name=job2" in txt
    assert "#SBATCH --time=02:30:00" in txt
    assert "#SBATCH --ntasks=2" in txt
    assert "#SBATCH --cpus-per-task=4" in txt
    assert "#SBATCH --mem=12.50GB" in txt
    assert "#SBATCH --partition=work" in txt
    assert "#SBATCH --account=acc" in txt
    assert "#SBATCH --qos=low" in txt
    assert "orca6" in txt


def test_render_pbs_qchem_chunk_mode():
    txt = render_pbs_qchem(job_name="water", chunk_size=5)
    assert "FILES_PER_JOB=5" in txt
    assert "qsub \"${pbsfile}\"" in txt
    assert "qchem -np ${NCPUS}" in txt


def test_render_slurm_orca_chunk_mode():
    txt = render_slurm_orca(job_name="orca_job", chunk_size=3)
    assert "FILES_PER_JOB=3" in txt
    assert "sbatch \"${sbatchfile}\"" in txt
    assert "files_to_run" in txt


def test_run_control_wrapper_includes_attempt_and_delete_rules():
    txt = render_pbs_qchem(job_name="job1")
    assert "._try.out" in txt  # log naming for attempts
    assert "skip_if_done" in txt
    assert "allow_delete_outputs" in txt
