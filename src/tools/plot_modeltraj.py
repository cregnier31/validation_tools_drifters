from lib_pyt.plot.class_plot_map import PlotMapCartopy as map2D
import xarray as xr
import pandas as pd
import numpy as np
import os

dateval = "20240104"
SYSTEM = "SMOC"
file = "/scratch/work/regnierc/WORK_lagrangian/DRIFT_SMOC/lagrangian_belenos/Parcels_Drift-HDCT_20240104-20240115/ZONE_180W180E-90S90N/20240104/Trajectories_SMOC_20240104_20240114_at_SVPloc_smag.zarr"
file = "/scratch/work/regnierc/WORK_lagrangian/DRIFT_SMOC/lagrangian_belenos/Parcels_Drift-HDCT_20240105-20240116/ZONE_180W180E-90S90N/20240105/Trajectories_SMOC_20240105_20240115_at_SVPloc_smag.zarr"
traj_lon_int = []
traj_lat_int = []
traj_time_int = []
traj_IDS_int = []
ds = xr.open_zarr(file)
traj_time_int.append(ds.time.values)
#lon = ~(np.isnan(ds.lon.values)
lon = ds.lon.values
lat = ds.lat.values
#traj_lon_int.append(ds.lon.values)
#traj_lat_int.append(ds.lat.values)
#traj_IDS_int.append(ds['label'].values)

#lon = traj_lon_int
#lat = traj_lat_int
print(np.array(lon))
print(np.array(lat))
#sys.exit(1)
dirout = "/home/smer/regnierc/code/validation_tools_drifters/imgs/"
map2d = map2D()
nameout = f"{SYSTEM}_Lagrangian_Trajectories_{dateval}"
map2d.plot_dot(np.array(lon), np.array(lat), color = 'red',
           alpha=0.9, marker='o', markersize=20,
           output=os.path.join(dirout, nameout),
           title=f'Trajectories {SYSTEM} {dateval}')
