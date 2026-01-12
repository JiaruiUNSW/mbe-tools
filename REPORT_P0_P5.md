# mbe-tools Progress Report (P0–P5)

This report summarizes the changes implemented across phases P0 through P5.

## P0 – Settings Loader
- Added hierarchical settings with precedence: env vars < `~/.config/mbe-tools/config.toml` < `./mbe.toml` < explicit path.
- Keys include commands/modules for Q-Chem/ORCA, scratch directory, and scheduler hints (queue/partition/account).
- CLI and loaders now share defaults via `load_settings`.

## P1 – Auto Program Detection & Metadata
- Parser auto-detects backend (`qchem`/`orca`) when `--program auto` is used.
- Path-aware metadata inference for method/basis/grid; emitted fields include `program_detected`, `status`, `error_reason`, and timing (CPU/wall seconds).

## P2 – Spatial Sampling Mode
- New centroid-based sampling (`--mode spatial`) to build compact fragment sets, with options `--prefer-special`, `--k-neighbors`, and `--start-index`.
- Ion/special-fragment retention supported during sampling.

## P3 – Connectivity Fragmentation & Labeling
- Connectivity-driven fragmentation using covalent radii + union-find.
- Automatic fragment labels for common motifs (water, methanol, ethanol, benzene, ions) and retention of special fragments.

## P4 – MBE Params & Input Tunables
- `MBEParams`/CLI expose explicit order selection (`--orders`), CP toggles, and scheme labels.
- Input builders support threshold/tolerance/SCF convergence (Q-Chem) and grid/SCF controls (ORCA) with CLI flags.

## P5 – Run-Control in HPC Templates
- PBS/Slurm templates wrap executions with a Python run-control harness.
- Features: regex confirmation on temp logs, retries with cleanup/sleep, optional failed-log alias, state recording to `.mbe_state.json`, and optional deletion (guarded by `allow_delete_outputs`).
- Control file discovery: `<input>.mbe.control.toml` or `mbe.control.toml`; strict mode rejects malformed control files.

## Tests
- Full suite (`python -m pytest`) currently passes (29 tests).
