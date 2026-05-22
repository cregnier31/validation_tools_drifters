from lib_pyt.plot.class_plot_map import PlotMapCartopy as map2D
import pandas as pd
import numpy as np
import os

dateval = "20251203"
file = "/utmp/smer/regnierc/WORK/DRIFT_GLO12/project_fidji_new_test/fidji_parcels_NbmonthsDrift-1_20251203-20260111/ZONE_180W180E-90S90N/DRIFTERS/Drifters_180W180E-90S90N_20251203_20251213.csv"
df = pd.read_csv(file, parse_dates=["time"])

lon = df['lon']
lat = df['lat']
dirout = "/home/smer/regnierc/code/validation_tools_drifters/imgs/"
map2d = map2D()
nameout = f"Positions_Drifters_{dateval}"
map2d.plot_dot(np.array(lon), np.array(lat), color = 'red',                                                                                                                                                    
           alpha=0.9, marker='o', markersize=20,
           output=os.path.join(dirout, nameout),
           title=f'15m Drifter positions  {dateval}')


print(df['lon'])
## Sort properly (important for trajectories)
#df = df.sort_values(["id", "time"])
#
#for drifter_id, group in df.groupby("id"):
#    plt.plot(group["lon"], group["lat"], linewidth=1)
#
