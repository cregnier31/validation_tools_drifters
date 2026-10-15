# Refactor audit and 28-day ensemble forecast plan

## Existing architecture
- `Lagrangian_tool_traj_compute.py` orchestrates seeding, data access and particle runs.
- `src/Functions.py` coordinates drifter retrieval, trajectory runs and graphics.
- `src/Drift_exec.py` contains the Parcels implementation (local NetCDF and xarray-backed remote variants).
- `src/Drift_Tools.py` implements trajectory/skill calculations (Liu, separation and related helpers).
- `Lagrangian_tool_lagrangian_scores.py` calculates/report-cards the scores after trajectories exist.
- `class1-comparisons/stats_fcst_io.py` provides the reference pattern for S3/Zarr enumeration, `s3fs.S3Map`, and lazy xarray opening; `tools.py:get_surface_ensemble_mean` computes means along `ens`.

## Refactoring candidates (not removed until call sites and notebook usage are tested)
- Duplicate imports and repeated `Age`, `SampleParticle`, and boundary/diffusion kernels inside `src/Drift_exec.py`. These are nested in individual runners, so consolidate only after numerical regression checks.
- Functions imported but not necessarily executed: audit static call sites in the three scripts and both notebooks. Imported names alone are not evidence that a function can safely be removed.
- Remove tracked `src/__pycache__/*.pyc`, `src/.ipynb_checkpoints/`, and `src/Drift_exec.py.bck` *after* reviewing reproducibility needs. Keep scientific figures unless explicitly superseded.
- `src/Functions.py` already writes particle **output** to Zarr; the new requirement is Zarr **model input**.
- `Drift_Tools.LiuScore2` should be tested for initial zero denominators and off-by-one accumulation before any change.

## Proposed processing flow
1. Discover stores for each forecast cycle from `*_RYYYYMMDD.zarr` names; resolve U and V grid-specific stores independently where needed.
2. Read `uo` and `vo` lazily using xarray and `fsspec`/`s3fs`, normalize grid coordinate names, align valid times and check units/masks.
3. Select 28-day lead window (lead 0 through day 28); calculate `mean('ens')` per velocity component *before* integration, at configurable depth.
4. Seed the same reference drifters for each forecast cycle, integrate with Parcels xarray FieldSet, store trajectories keyed by run date and lead time.
5. Match observed drifter positions by **valid time**, enforce common coverage, then call existing separation distance, Liu and Pearson score calculations without altering definitions.
6. Save per-cycle/per-lead outputs and aggregate/report with counts of valid drifter pairs.

## Essential tests
- Synthetic 2-member constant U/V ensemble yields correct mean velocity and trajectory.
- Multi-cycle forecasts keep run and lead coordinates distinct, including overlapping validity periods.
- Missing velocity fields, mismatched U/V time axes, and incomplete 28-day windows fail clearly.
- Curvilinear NEMO U/V grids, masks, longitude wrapping, depth choice, velocity units and Parcels interoperability.
- NetCDF regression: identical velocities reproduce previous trajectories/scores within an agreed tolerance.
- Correctly compare `trajectory(mean(U,V))` versus `mean(trajectory(member))`; they are distinct diagnostics.

## Status
The `src/forecast_inputs.py` adapter has been added on the refactor branch as a starting point, **not wired to the runner**. No existing scores or numerical kernels have been deleted or changed. No execution against real model stores has been performed.
