# GLO12_025: two-environment conservative remapping

**Do not install xESMF or ESMF in the working Parcels environment.**

## 1. Create a dedicated preprocessing environment (once)

```bash
conda create -n glo12_regrid --override-channels -c conda-forge \
  python=3.11 xesmf esmpy xarray scipy netcdf4 zarr s3fs -y
```

## 2. Pull the implementation

```bash
git pull --ff-only origin feature/gloens-s3-zarr-discovery
```

## 3. Generate the weights (once per unchanged source grid)

Use the same cache directory as the Parcels runner. The runtime uses the configured work directory directly, without an `AdvectionRK4/` subdirectory.

```bash
conda run -n glo12_regrid python scripts/generate_glo12_025_weights.py \
  --store s3://oceanbench-bucket/public/GLO12/20240103.zarr \
  --endpoint-url https://s3.waw3-1.cloudferro.com \
  --cache-dir /scratch/cregnier/WORK_lagrangian/glo12_025_weights
```

This step alone imports xESMF/ESMF. The generated NetCDF sparse weight file is keyed by the exact source and target coordinate arrays. Generation may be memory-intensive for a global grid.

## 4. Run GLO12_025 with the existing Parcels environment

```bash
bash Launch_Parcels.sh 20240103 20240103 GLO12_025 \
  --console yes --lead FCST --scheme AdvectionRK4 \
  --drift-days 10 --delete-failed-particles
```

After a successful single-cycle test, extend the dates. The Parcels environment needs SciPy, xarray and a NetCDF reader, but **not** xESMF/ESMF. A missing weight file raises an explicit error with its expected path.

The runtime loads ESMF's sparse row/column/weight arrays, excludes missing source values, and masks target cells with less than 50% valid contributing area. The method uses **first-order conservative** weights, not bilinear sampling. The 1/4-degree target is a regular geographic grid, not the native curvilinear GLOENS grid.

## 5. Validate scientifically

Check the target grid and land/coastal masking; compare particle survival and the paired drifter cohort. The coarsened GLO12 field is a spatially filtered high-resolution forecast, not a model dynamically rerun at 1/4 degree. Keep launch dates, observations, time steps and verification definitions identical when comparing GLO12, GLO12_025 and GLOENS.

**Status:** committed, not yet executed or validated on the HPC cluster.
