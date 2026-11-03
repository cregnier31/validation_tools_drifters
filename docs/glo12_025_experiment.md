# GLO12_025 Lagrangian resolution-sensitivity experiment

Run the original GLO12 forecast and the conservative 1/4-degree variant independently:

```bash
bash Launch_Parcels.sh 20240103 20240131 GLO12 --console yes --lead FCST --scheme AdvectionRK4 --drift-days 10 --delete-failed-particles
bash Launch_Parcels.sh 20240103 20240131 GLO12_025 --console yes --lead FCST --scheme AdvectionRK4 --drift-days 10 --delete-failed-particles
```

The second command reads the **same weekly GLO12 Zarr stores** and retains exactly ten daily input fields per initialization. Before Parcels integration, it uses xESMF/ESMF `conservative_normed` area-overlap remapping on each eastward and northward velocity component. Missing source values are ignored with a 50% valid-area threshold (`na_thres=0.5`). The target is a regular 0.25-degree grid with cells wholly within the source's latitude/longitude coverage. Source and target coordinate arrays determine the sparse remapping weights, which are cached under `DIR_WORK/glo12_025_weights/`.

**Dependency:** install `xesmf` and an ESMF backend (`esmpy`) in the **Parcels Python environment**, preferably from conda-forge. This is an opt-in dependency: baseline GLO12, GLONET and GLOENS runs do not import it.

Outputs for `GLO12_025` have their own system name in the trajectory directory, filenames, statistics and plots. Do not reuse native GLO12 trajectories for this experiment.

**Interpretation:** the experiment spatially filters GLO12 velocities; it does not turn GLO12 into a dynamically native quarter-degree ocean model. To compare GLO12, GLO12_025 and GLOENS fairly, pair the same initialization dates, observed drifters, valid-particle cohort, integration parameters and score definitions. The original runner's end-of-window persistence extension is unchanged.

**Validation before production:** check ESMF installation, first-run memory and weight generation, coastal masks, and Parcels behavior on a short weekly cycle. This implementation has not been executed on the HPC cluster.
