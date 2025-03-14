import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import matplotlib
import cartopy
import numpy as np
from datetime import timedelta,datetime
import netCDF4 as nc




def comparative_maps(MAPS,AXS_TITLES,cell_size,grid,fig_title,fig_name,VMIN,VMAX):
    if False: #for i in range(len(MAPS)):
        data = MAPS[i]
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)
        
        limit = 2 * std_dev
        MAPS[i][np.abs(data - mean) > limit] = np.nan
    
        MIN_val = []
        MAX_val = []
        for i in range(len(MAPS)):
            MIN_val.append(np.nanmin(MAPS[i]))#     np.ma.min(np.ma.masked_array(MAPS[i],   np.logical_or(np.isinf(MAPS[i]) , np.isnan(MAPS[i])  ) )))
            MAX_val.append(np.nanmax(MAPS[i]))#    np.ma.max(np.ma.masked_array(MAPS[i],   np.logical_or(np.isinf(MAPS[i]) , np.isnan(MAPS[i])  ) )))

        VMIN = np.min(MIN_val)
        VMAX = np.max(MAX_val)


    fig,axs = plt.subplots(figsize = (4*len(MAPS),6),ncols=len(MAPS),nrows = 1,subplot_kw={'projection': ccrs.PlateCarree()})
    if len(MAPS) >1:
        for i in range(len(MAPS)):
            print(i)
            #for j in range(2):
            axs[i].coastlines()
            axs[i].add_feature(cfeature.LAND)
            gl = axs[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.xlabels_top = False
            gl.ylabels_right = False
            fig.suptitle(fig_title,fontsize = 12)
            axs[i].set_title(AXS_TITLES[i]) #      (MODELS[i]+' : \n'+str(prop_tot[i])+"%")
            cb = axs[i].pcolormesh(grid[0]+cell_size,grid[1]+cell_size,MAPS[i], vmin = VMIN,vmax = VMAX,cmap ='jet' )
            axs[i].add_feature(cfeature.LAND, zorder=5)
            #fig.colorbar(cb, ax=axs.ravel().tolist(),orientation = 'horizontal',label="%")
    else:
        axs.coastlines()
        axs.add_feature(cfeature.LAND)
        gl = axs.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabels_top = False
        gl.ylabels_right = False
        fig.suptitle(fig_title,fontsize = 12)
        axs.set_title(AXS_TITLES[0])
        cb = axs.pcolormesh(grid[0]+cell_size,grid[1]+cell_size,MAPS[0],vmin = VMIN,vmax = VMAX,cmap ='jet')
        axs.add_feature(cfeature.LAND, zorder=5)



    fig.colorbar(cb,orientation = 'horizontal')#,label="%")
    #plt.savefig(DIR_OUT+'PLOT/map_proportion_contour.png',dpi = 400)
    plt.savefig(fig_name,dpi = 400)
    print(VMIN)
    print(VMAX)


def Compute_2D_MAPS(param,drifter_lon,drifter_lat,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,cell_size):
    grid = np.meshgrid(np.linspace(LON_MIN,LON_MAX,int((LON_MAX - LON_MIN )/cell_size)+ 1),np.linspace(LAT_MIN,LAT_MAX,int((LAT_MAX - LAT_MIN )/cell_size)+ 1))
    MAP = np.zeros(grid[0].shape)*np.nan
    NOBS = np.zeros(grid[0].shape)
    
    print(param.flatten())

    traj_drifter_lon = drifter_lon.flatten()
    traj_drifter_lat = drifter_lat.flatten()
    for i in range (grid[0].shape[0]-1):
        for j in range (grid[0].shape[1]-1):
            indice = np.where(np.array((grid[0][i,j]< traj_drifter_lon)*(traj_drifter_lon<grid[0][i,j]+cell_size)*(grid[1][i,j]< traj_drifter_lat)*(traj_drifter_lat<grid[1][i,j]+cell_size)))
            if indice[0].shape[0]!= 0:
                MAP[i,j] = np.nanmean(param.flatten()[indice[0]])
                NOBS[i,j] = indice[0].shape[0]
    #ff=${MODEL_INPUT}_${FREQ}_${d1}_${d1}_${g}_R${dr}.nc
    return(MAP,NOBS,grid)


def compute_scores_TS(score,dates,lons,lats,date_start,date_end,zones,bool_global=True):
    score_flat = score.flatten()
    date_flat = dates.flatten()
    print(date_flat)
    format = '%Y-%m-%d'
    dstart = date_start#  datetime.strptime(date_start, format)
    dend = date_end #datetime.strptime(date_end, format)

    keysList = list(zones.keys())
    current_day = dstart
    if bool_global == False:
        scores_vector = []
        dates_vector = []
        nobs_vector = []
        while current_day <= dend:
            print(current_day)
            indice = np.where(np.array( date_flat  == current_day.strftime('%Y-%m-%d')))
            if indice[0].shape[0]!= 0:
                print(current_day)
                dum = np.nanmean(score_flat[indice[0]])
                scores_vector.append(dum)
                dates_vector.append(current_day)
                nobs_vector.append(len(indice[0]))
            current_day += timedelta(days=1)
    else:
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        print(np.shape(lats_flat))
        print(np.shape(date_flat))
        scores_vector = [[]]
        dates_vector = [[]]
        nobs_vector = [[]]
        for zone in zones:
            scores_vector.append([])
            dates_vector.append([])
            nobs_vector.append([])
        while current_day <= dend:
            print(current_day)
            bool0 = np.array(date_flat  == current_day.strftime('%Y-%m-%d'))
            indice = np.where(bool0)
            if indice[0].shape[0]!=0:
                dum = np.nanmean(score_flat[indice[0]])
                scores_vector[0].append(dum)
                dates_vector[0].append(current_day)
                nobs_vector[0].append(len(indice[0]))
            for ii in range(len(keysList)):
                bool1 = np.array(lats_flat[indice]  <= zones[keysList[ii]]['lat_max']) # 'lon_min': -19, 'lon_max': 5, 'lat_min': 26, 'lat_max': 56 
                bool2 = np.array(lats_flat[indice]  >= zones[keysList[ii]]['lat_min']) 
                bool3 = np.array(lons_flat[indice]  <= zones[keysList[ii]]['lon_max'])
                bool4 = np.array(lons_flat[indice]  >= zones[keysList[ii]]['lon_min'])        
                print(np.shape(bool1))
                print(np.shape(bool2))
                print(np.shape(bool3))
                print(np.shape(bool4))
                
                selecting = np.where(bool1*bool2*bool3*bool4)
                if indice[0].shape[0]!=0:
                    dum = np.nanmean(score_flat[selecting[0]])
                    scores_vector[ii+1].append(dum)
                    dates_vector[ii+1].append(current_day)
                    nobs_vector[ii+1].append(len(selecting[0]))
            current_day += timedelta(days=1)
    print(scores_vector)
    print(dates_vector)
    print(nobs_vector)
    return(scores_vector,dates_vector,nobs_vector)

def save_TS_nc(scores_vector,dates_vector,nobs_vector,nc_file):
    # Create the ,nc file
    dates_vector = np.array(dates_vector, dtype='datetime64[ns]')
    ncfile = nc.Dataset(nc_file, 'w', format='NETCDF4')
 
    # Create the dimensions of the  NetCDF file
    n_zones = ncfile.createDimension('zone', len(scores_vector))#.shape[0])
    n_dates = ncfile.createDimension('time', len(scores_vector[0]))#.shape[1])

    # Create the main variable 
    scores = ncfile.createVariable('scores', np.float32, ('zone', 'time'))
    nobs = ncfile.createVariable('nobs', np.float32, ('zone', 'time'))
    #dates = ncfile.createVariable('dates',np.datetime64, ('zone', 'time'))
    dates = ncfile.createVariable('dates', np.int64, ('zone', 'time'))

     # Allocate the main variable
    scores[:,:] = scores_vector#DAta
    nobs[:,:] = nobs_vector
    # Convert datetime.datetime to numerical representation suitable for NetCDF
    # For example, using Unix timestamps
    dates_vector_unix = np.array([(dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for dt in dates_vector])
    dates[:, :] = dates_vector_unix
    
    
    
    #dates[:,:] = dates_vector
    # Close the NetCDF file
    ncfile.close()
    print(f"NetCDF file '{nc_file}' created successfully.")




def plot_scores_TS(scores_vector,dates_vector,nobs_vector,zones,fig_title,fig_label,dir_out,score_name,bool_multi_zones = True):
    print(len(scores_vector))
    list_colors = ['r','m','c','g','k']
    if bool_multi_zones == False:
        dates = dates_vector
        x = scores_vector
        y = nobs_vector
        sizes =  np.linspace(10, 100,len(scores_vector)) #  
        # Tracer le graphique
        plt.figure(figsize=(10, 6))
        plt.scatter(dates, x, c=y, s=sizes, cmap='viridis', alpha=0.5)
        plt.colorbar(label=score_name)  # Légende pour la couleur
        
        plt.xlabel('dates')
        plt.ylabel(score_name)#'score')
        plt.title(fig_title)# + f'( --- The size of points <-> Nobs')
        score_ab = score_name.split()[0] +'_'+ score_name.split()[1]
        fig_name = dir_out +'PLOT/TS_' + 'Gobal' + '_' + score_ab + '.png' 
        plt.savefig(fig_name,dpi = 400)
    else:
        dates_all = []
        data_all = []
        nobs_all = []
        dates = dates_vector[0]
        x = scores_vector[0]
        y = nobs_vector[0]
        NOBSMAX= np.nanmax(y)
        print(NOBSMAX)
        sizes = 100*(y/NOBSMAX)# np.linspace(0, 100,len(x)) #
        print(y)
        print(sizes)
        
        # append for scatterplot
        #data_all.append(x)
        #dates_all.append(dates)

        
        # Tracer le graphique
        plt.figure(figsize=(10, 6))
        #plt.scatter(dates, x, c=y, s=sizes, cmap='viridis', alpha=0.5)
        #plt.colorbar(label=score_name)  # Légende pour la couleur
        plt.plot(dates, x, linestyle='--', color='b',label='Global')
        #plt.xlabel('dates')
        #plt.ylabel(score_name)#'score')
        # plt.title(fig_title + f'( --- The size of points <-> Nobs)')
        score_ab = score_name.split()[0] +'_'+ score_name.split()[1]
        #fig_name = dir_out +'PLOT/TS_' + 'Gobal' + '_' + score_ab + '.png'
        #plt.savefig(fig_name,dpi = 400)
        keysList = list(zones.keys())
        for ii in range(len(zones)):
            print(keysList[ii])
            dates = dates_vector[ii+1]
            x = scores_vector[ii+1]
            y = nobs_vector[ii+1]
            
            #sizes = 100*(y/NOBSMAX) #  np.linspace(0, 100*ZONENOBSMAX/NOBSMAX,len(x)) 
            #hold on;
            #plt.figure(figsize=(10, 6))
            data_all.append(x)
            dates_all.append(dates)
            nobs_all.append(y)
            
            #plt.scatter(dates, x, c=y, s=sizes, cmap='viridis', alpha=0.5)
            plt.plot(dates, x, linestyle='--', color= list_colors[ii], label = keysList[ii])
            #plt.colorbar(label=score_name)  # Légende pour la couleur
    
    x = np.array([item for items in data_all for item in items])
    d = np.array([item for items in dates_all for item in items])
    y = np.array([item for items in nobs_all for item in items])
    
    NOBSMAX =  np.nanmax(y)
    y = 35*y/NOBSMAX#   [iii] = 34
    sizes = y# 100*(y/NOBSMAX)
    #print(np.shape(x))
    #print(np.shape(y))
    plt.scatter(d, x, c=y, s=sizes, cmap='jet', alpha=0.5)
    plt.colorbar(label='Number of observations per zone')  # Légende pour la couleur

    plt.legend()
    plt.xlabel('dates')
    plt.ylabel(score_name)#'score')
    plt.title(fig_title)# + f'( --- The size of points <-> Nobs)')
    plt.grid(True)
    fig_name = dir_out +'PLOT/TS_' + fig_label + '_' + score_ab + '.png'
    plt.savefig(fig_name,dpi = 400)



def save_ncfile_maps(DAta,ncfile_name,var_name,lat_min,lat_max,lon_min,lon_max,box_size,data_unit):   
    # Create the ,nc file
    ncfile = nc.Dataset(ncfile_name, 'w', format='NETCDF4')


    # Create the dimensions of the  NetCDF file
    lat_dim = ncfile.createDimension('lat', DAta.shape[0])
    lon_dim = ncfile.createDimension('lon', DAta.shape[1])

    # Create the variables lon/lat
    latitudes = ncfile.createVariable('lat', np.float32, ('lat',))
    longitudes = ncfile.createVariable('lon', np.float32, ('lon',))

    # Create the main variable 
    DAta_var = ncfile.createVariable(var_name, np.float32, ('lat', 'lon'))

    # Define the attributes of fimensions variables 
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    DAta_var.units = data_unit  
    # Define the values of dimension  variables 
    latitudes[:] = np.linspace(lat_min,lat_max,DAta.shape[0])
    longitudes[:] = np.linspace(lon_min,lon_max,DAta.shape[1])
    
    # Allocae the main variable
    DAta_var[:,:] = DAta
    
    # Close the NetCDF file
    ncfile.close()

    print(f"NetCDF file '{ncfile_name}' created successfully.")

#def save_ncfile_maps(DAta,ncfile_name,var_name,lat_min,lat_max,lon_min,lon_max,data_unit):

