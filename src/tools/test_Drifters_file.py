import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


file_path = '/utmp/smer/regnierc/DATA/CURRENTS/NRT/INSITU_GLO_UV_NRT_OBSERVATIONS_013_048/drifter_filt/2024/'

nc_files_jan = glob.glob(file_path + 'GL_TS_DC_202401*.nc')
nc_files_feb = glob.glob(file_path + 'GL_TS_DC_202402*.nc')
nc_files_mar = glob.glob(file_path + 'GL_TS_DC_202403*.nc')
nc_files_apr = glob.glob(file_path + 'GL_TS_DC_202404*.nc')
nc_files_may = glob.glob(file_path + 'GL_TS_DC_202405*.nc')
nc_files_jun = glob.glob(file_path + 'GL_TS_DC_202406*.nc')


def create_ncfile(nc_files):
    datasets = []
    for file in nc_files:
        ds = xr.open_dataset(file)
        if 'POSITION' in ds.dims:
            ds = ds.drop_dims('POSITION') 
        if 'STRING64' in ds.dims:
             ds = ds.drop_dims('STRING64')
        
        #min_position_size = min([ds.sizes['POSITION'] for ds in datasets])
        #datasets_trimmed = [ds.isel(POSITION=slice(0, min_position_size)) for ds in datasets]
        #combined_ds = xr.concat(datasets_trimmed, dim='TIME')
        #combined_ds = xr.concat(datasets, dim='TIME', join='outer')
        ds['LATITUDE'] = xr.DataArray(ds['LATITUDE'].values, dims='TIME', coords={'TIME': ds['TIME']})
        ds['LONGITUDE'] = xr.DataArray(ds['LONGITUDE'].values, dims='TIME', coords={'TIME': ds['TIME']})
        ds = ds.drop_dims(['LATITUDE', 'LONGITUDE'])
        datasets.append(ds)

        combined_ds = xr.concat(datasets, dim='TIME')  
    return(combined_ds)

ds_jan = create_ncfile(nc_files_jan)
ds_feb = create_ncfile(nc_files_feb)
ds_mar = create_ncfile(nc_files_mar)
ds_apr = create_ncfile(nc_files_apr)
ds_may = create_ncfile(nc_files_may)
ds_jun = create_ncfile(nc_files_jun)



print(len(ds_jan['LATITUDE'].data))
print(len(ds_may['LATITUDE'].data))


variable = 'LATITUDE'
Min_vals_lat = [np.nanmin(ds_jan[variable].data), np.nanmin(ds_feb[variable].data), np.nanmin(ds_mar[variable].data), np.nanmin(ds_apr[variable].data), np.nanmin(ds_may[variable].data),np.nanmin(ds_jun[variable].data)]
Max_vals_lat = [np.nanmax(ds_jan[variable].data), np.nanmax(ds_feb[variable].data), np.nanmax(ds_mar[variable].data), np.nanmax(ds_apr[variable].data), np.nanmax(ds_may[variable].data),np.nanmax(ds_jun[variable].data)]

 
variable = 'LONGITUDE'
Min_vals_lon = [np.nanmin(ds_jan[variable].data), np.nanmin(ds_feb[variable].data), np.nanmin(ds_mar[variable].data), np.nanmin(ds_apr[variable].data), np.nanmin(ds_may[variable].data),np.nanmin(ds_jun[variable].data)]
Max_vals_lon = [np.nanmax(ds_jan[variable].data), np.nanmax(ds_feb[variable].data), np.nanmax(ds_mar[variable].data), np.nanmax(ds_apr[variable].data), np.nanmax(ds_may[variable].data),np.nanmax(ds_jun[variable].data)]


print(Min_vals_lon)
print(Max_vals_lon)

print(Min_vals_lat)
print(Max_vals_lat)



fig_dirs = '/home/smer/sval001/WORK/DRIFT/test_figs/'



plt.figure(figsize=(10, 6))
#plt.subplot(3,2,1)
plt.plot( ds_jan['LATITUDE'].data[1:-1:6], ds_jan['LONGITUDE'].data[1:-1:6])
plt.title('January')
plt.savefig(fig_dirs + 'jan.png')#, dpi=300)  # Save as a PNG file with 300 DPI

plt.figure(figsize=(10, 6))
plt.plot( ds_apr['LATITUDE'].data[1:-1:6], ds_apr['LONGITUDE'].data[1:-1:6])
plt.title('April')
plt.savefig(fig_dirs + 'apr.png')#, dpi=300)  # Save as a PNG file with 300 DPI



plt.figure(figsize=(10, 6))
#plt.subplot(3,2,3)
plt.plot( ds_feb['LATITUDE'].data[1:-1:6])
#, ds_feb['LONGITUDE'].data[1:-1:6])
plt.title('February')
plt.savefig(fig_dirs + 'feb_lat.png')#, dpi=300)  # Save as a PNG file with 300 DPI


plt.figure(figsize=(10, 6))
plt.plot( ds_feb['LONGITUDE'].data[1:-1:6])
plt.title('February')
plt.savefig(fig_dirs + 'feb_lon.png')#, dpi=300)  # Save as a PNG file with 300 DPI




plt.figure(figsize=(10, 6))
#plt.subplot(3,2,4)
plt.plot( ds_may['LATITUDE'].data[1:-1:6], ds_may['LONGITUDE'].data[1:-1:6])
plt.title('May')
plt.savefig(fig_dirs + 'may.png')  # Save as a PNG file with 300 DPI



plt.figure(figsize=(10, 6))
#plt.subplot(3,2,5)
plt.plot( ds_mar['LATITUDE'].data[1:-1:6], ds_mar['LONGITUDE'].data[1:-1:6])
plt.title('March')
plt.savefig(fig_dirs + 'mar.png')#, dpi=300)  # Save as a PNG file with 300 DPI



plt.figure(figsize=(10, 6))
#plt.subplot(3,2,6)
plt.plot( ds_jun['LATITUDE'].data[1:-1:6], ds_jun['LONGITUDE'].data[1:-1:6])
plt.title('June')
plt.savefig(fig_dirs + 'jun.png')#, dpi=300)  # Save as a PNG file with 300 DPI


# Adjust layout for better display
# plt.tight_layout()

# Save the figure to a file
#plt.savefig('my_figure.png', dpi=300)  # Save as a PNG file with 300 DPI

# Show the figure
#)



#print(ds_feb)
#print(ds_mar)
#print(ds_apr)
#print(ds_may)
#print(ds_jun)


plt.figure(figsize=(10, 6))
data = ds_jan['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('January latitude')
plt.savefig(fig_dirs + 'jan_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_jan['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('January longitude')
plt.savefig(fig_dirs + 'jan_lon.png')#


plt.figure(figsize=(10, 6))
data = ds_feb['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('february latitude')
plt.savefig(fig_dirs + 'feb_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_feb['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('February longitude')
plt.savefig(fig_dirs + 'feb_lon.png')#



plt.figure(figsize=(10, 6))
data = ds_mar['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('March latitude')
plt.savefig(fig_dirs + 'mar_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_mar['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('March longitude')
plt.savefig(fig_dirs + 'mar_lon.png')#




plt.figure(figsize=(10, 6))
data = ds_apr['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('April latitude')
plt.savefig(fig_dirs + 'apr_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_apr['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('April longitude')
plt.savefig(fig_dirs + 'apr_lon.png')#


plt.figure(figsize=(10, 6))
data = ds_may['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('May latitude')
plt.savefig(fig_dirs + 'may_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_may['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('May longitude')
plt.savefig(fig_dirs + 'may_lon.png')#







plt.figure(figsize=(10, 6))
data = ds_jun['LATITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('June latitude')
plt.savefig(fig_dirs + 'jun_lat.png')#

plt.figure(figsize=(10, 6))
data = ds_jun['LONGITUDE']
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('June longitude')
plt.savefig(fig_dirs + 'jun_lon.png')#







