'''
Calculation of Persistent Positive Anomalies (PPAs) in 500hPa geopotential heights for Canada
Algorithm adapted from Sharma et al. 2022

Sharma, A.R., Jain, P., Abatzoglou, J.T. and Flannigan, M., 2022. 
Persistent Positive Anomalies in Geopotential Heights Promote Wildfires in Western North America. 
Journal of Climate, 35(19), pp.6469-6486.

https://journals.ametsoc.org/view/journals/clim/35/19/JCLI-D-21-0926.1.xml

data needed:
500 hPa geotential heights for given year
500 hPa geopotential height climatology (both mean and standard deviation, for 1991-2020 or other appropriate period)
Area of each grid cell 

Piyush Jain
Dec 10, 2023
'''

# imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import analysis.PPA_identify as PPA

# reload packages as developed.
import importlib
importlib.reload(PPA)


#########################################################
# parameters to set - original dataset

'''
years = np.arange(1991, 2023+1)
year = 2023
area = "NA"
sigma_threshold = 1. # standard deviations for climatology
size_threshold = 100. # number of grid cells
duration_threshold = 5 # days
area_threshold = 100000. # units = km^2
study_extent = [-150.0, 40.0, -50.0, 75.0] # might be smaller than full extent of loaded data
path_Z500 = "/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential/"
path_output = "/Users/piyush/RESEARCH/PROJECTS/2023_fire_season/PPA_results/"
show_test_plots = False

#########################################################
# generated using commmand line CDO
# cdo gridarea geopotential_500_NA_1991_10_daily.mean.nc gridarea.nc
gridarea = xr.open_dataset(path_Z500 + "gridarea_"+area+".nc")

# set paths to climatology
Z500_filepath = path_Z500 + 'geopotential_500_NA_{0}_daily.mean.nc'
Z500_era5_clim_filepath = path_Z500 + "climatology_daily/geopotential_500_NA_climatology_daily.mean.nc"
Z500_sigma_era5_clim_filepath = path_Z500 + "climatology_daily_4wk/geopotential_500_sigma_NA_climatology_daily_4wk.mean.nc"

'''

#########################################################
# parameters to set - climxp dataset

years = np.arange(1950, 2023+1)
year = 2023
area = "NA"
sigma_threshold = 1. # standard deviations for climatology
size_threshold = 100. # number of grid cells
duration_threshold = 5 # days
area_threshold = 100000. # units = km^2
study_extent = [-150.0, 40.0, -50.0, 75.0] # might be smaller than full extent of loaded data
path_Z500 = "/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential_climxp/"
path_output = "/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/output_climxp/"
show_test_plots = False

# set paths to main data and detrended climatology - detrended climatology 
Z500_filepath = path_Z500 + 'Z500_NA_annual/geopotential_500_NA_{0}_daily.mean.nc' # non-detrended
#Z500_filepath = path_Z500 + 'Z500_NA_DRA_annual/geopotential_500_NA_detrended_{0}_daily.mean.nc' # detrended
Z500_era5_clim_filepath = path_Z500 + "climatology_daily_4wk/geopotential_500_NA_DRA_climatology_daily_4wk.mean.nc"
Z500_sigma_era5_clim_filepath = path_Z500 + "climatology_daily_4wk/geopotential_500_sigma_NA_DRA_climatology_daily_4wk.mean.nc"

#########################################################
# CDO gridarea for climxp data doesn't work:
#nc1 = xr.open_dataset("/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential/geopotential_500_NA_2023_10_daily.mean.nc")
#nc2 = xr.open_dataset("/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential_climxp/Z500_NA_annual/geopotential_500_NA_2023_daily.mean.nc")
#nc3 = nc2[["time", "lon", "lat", "z500"]]
#nc3 = nc3.transpose('time', 'lon', 'lat')
#nc3.to_netcdf("/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential_climxp/Z500_NA_annual/gridarea_input.nc")
# cdo gridarea gridarea_input.nc gridarea.nc

# instead do this in R as work around using terra::cellSize
# test = rast("/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential_climxp/Z500_NA_annual/geopotential_500_NA_2023_daily.mean.nc")
# gridarea = cellSize(test)
# writeCDF(gridarea, "/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential_climxp/Z500_NA_annual/gridarea.nc", overwrite=T)

gridarea = xr.open_dataset(path_Z500 + "Z500_NA_annual/gridarea.nc").rename({'gridarea':'cell_area'})
gridarea["cell_area"] = gridarea.cell_area.squeeze()


#########################################################
# MAIN PPA ALGORITHM
#########################################################

#########################################################
# Geopotential heights - LOAD DATA

for year in years:

    print("===============================================================================")
    print("PROCESSING YEAR " + str(year))
    print("===============================================================================")


    #Z500 = xr.open_mfdataset(path_Z500 + "geopotential_500_NA_"+str(year)+"_*_daily.mean.nc", decode_times=True)
    Z500 = xr.open_dataset(Z500_filepath.format(year)).rename({'z500':'z', 'lon':'longitude', 'lat':'latitude'})
    # 2023 needs padding since not complete year
    if (year == 2023 and Z500.time.shape < 365):
        Z500 = Z500.pad(time=(0,365-Z500.z.shape[0]))
    Z500 = Z500.sel(time=~((Z500.time.dt.month == 2) & (Z500.time.dt.day == 29)))
    Z500 = Z500.rolling(time=5, center=True).mean()
    Z500 /= 9.80665

    # climatology mean and standard deviation
    #Z500_era5_clim  = xr.open_dataset(path_Z500 + "climatology_daily/geopotential_500_NA_climatology_daily.mean.nc", decode_times=True)
    Z500_era5_clim = xr.open_dataset(Z500_era5_clim_filepath, decode_times=True).rename({'z500':'z', 'lon':'longitude', 'lat':'latitude'})
    Z500_era5_clim = Z500_era5_clim.rolling(time=5, center=True).mean()
    # put everything in same units
    Z500_era5_clim /= 9.80665

    #Z500_sigma_era5_clim  = xr.open_dataset(path_Z500 + "climatology_daily_4wk/geopotential_500_sigma_NA_climatology_daily_4wk.mean.nc", decode_times=True)
    Z500_sigma_era5_clim = xr.open_dataset(Z500_sigma_era5_clim_filepath, decode_times=True).rename({'z500':'z', 'lon':'longitude', 'lat':'latitude'})
    Z500_sigma_era5_clim = Z500_sigma_era5_clim.rolling(time=5, center=True).mean()
    # put everything in same units
    Z500_sigma_era5_clim /= 9.80665

    # anomalies
    Z500_anom = Z500.copy(deep=True)
    Z500_anom.z.values = (Z500_anom.z.values - Z500_era5_clim.z.values)

    doy_test = 180

    if show_test_plots:
        fig = plt.figure()
        Z500_anom.z[doy_test,:,:].plot()
        plt.show(block=False)

    # get extent for later - not actually used...
    #extent = PPA.get_extent(Z500_anom)

    ################################################################################
    # use amplitude threshold and create mask of potential PPA grid cells.
    # thresholding including latitudinal adjustment (which is strictly not necessary)

    Z500_anom_mask, PPA_mask = PPA.mask_Z500_anom_by_threshold(Z500_anom, Z500_sigma_era5_clim, sigma_threshold)
    PPA.print_PPA_props(PPA_mask, duration_threshold)

    # test plot
    if show_test_plots:
        fig = plt.figure()
        Z500_anom_mask.z[doy_test,:,:].plot()
        plt.show(block=False)

    ################################################################################
    # mask by minimum duration criterion
    PPA_mask = PPA.mask_array_by_duration(PPA_mask, duration_threshold)
    PPA.print_PPA_props(PPA_mask, duration_threshold)

    # test plot
    if show_test_plots:
        Z500_anom_mask = PPA.mask_Z500_anom_by_PPA_mask(Z500_anom, PPA_mask)
        fig = plt.figure()
        Z500_anom_mask.z[doy_test,:,:].plot()
        plt.show(block=False)


    # # back to xarray
    # PPA_mask_xr = Z500.copy(deep=True)
    # PPA_mask_xr.z.values = PPA_mask

    ################################################################################
    # mask by minimum area criterion

    PPA_mask = PPA.mask_by_area(PPA_mask, gridarea, area_threshold)
    PPA.print_PPA_props(PPA_mask, duration_threshold)

    # need to mask by duration again to ensure masking by area hasn't led to events
    # shorter than duration_threshold
    PPA_mask = PPA.mask_array_by_duration(PPA_mask, duration_threshold)
    PPA.print_PPA_props(PPA_mask, duration_threshold)

    # test plot
    if show_test_plots:
        Z500_anom_mask = PPA.mask_Z500_anom_by_PPA_mask(Z500_anom, PPA_mask)
        fig = plt.figure()
        Z500_anom_mask.z[doy_test,:,:].plot()
        plt.show(block=False)


    ################################################################################
    # create final xarray objects for plotting etc...

    # to do: keep all events that intereect with two following criteria
    # optional: filter by dates to only include events that overlap with given time period
    # filter by season - ie. event should overlap with season window
    # requires labelling and keeping all events that intersect with study period and study area.

    # # back to xarray
    PPA_mask_xr = Z500.copy(deep=True)
    PPA_mask_xr.z.values = PPA_mask

    # simpler filtering - just cut by date and study area. 
    filter_months = [4,5,6,7,8,9,10]
    PPA_mask_filtered_xr = PPA_mask_xr.sel(time=PPA_mask_xr.time.dt.month.isin(filter_months))
    Z500_anom_filtered = Z500_anom.copy(deep=True)
    Z500_anom_filtered = Z500_anom_filtered.sel(time=Z500_anom_filtered.time.dt.month.isin(filter_months))

    min_lon = study_extent[0] 
    min_lat = study_extent[1]  
    max_lon = study_extent[2] 
    max_lat = study_extent[3]  
    PPA_mask_filtered_xr = PPA_mask_filtered_xr.sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon))
    Z500_anom_filtered = Z500_anom_filtered.sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon))

    # labelling of filtered events
    PPA.print_PPA_props(PPA_mask_filtered_xr.z.values, duration_threshold)
    PPA_mask_labels = PPA.label_events(PPA_mask_filtered_xr.z.values)
    PPA_mask_labels_xr = PPA_mask_filtered_xr.copy(deep=True)
    PPA_mask_labels_xr.z.values = PPA_mask_labels

    # Z500_anom filtered (study period and study area) and PPA masked
    Z500_anom_filtered_masked = PPA.mask_Z500_anom_by_PPA_mask(Z500_anom_filtered, PPA_mask_filtered_xr.z.values)

    if show_test_plots:
        fig = plt.figure()
        Z500_anom_filtered.z[0,:,:].plot()
        plt.show(block=False)

        fig = plt.figure()
        Z500_anom_filtered_masked.z[0,:,:].plot()
        plt.show(block=False)

    # summary, number of PPA days per year
    PPA_days = PPA_mask_filtered_xr.sum(dim=["time"], skipna=True)

    # map these and save for reference
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # generate a basemap with country borders, oceans and coastlines
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='dotted')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
    color='gray', alpha=0.5)
    gl.xlabels_top = False
    gl.ylabels_right = False
    #cmap = plt.cm.get_cmap('Spectral_r', 8)    # 11 discrete colors
    PPA_days.z.plot(transform=ccrs.PlateCarree(), cmap="Spectral_r", cbar_kwargs={'label': "days"})
    plt.title("Number PPA days for " + str(year) + " (April-October)")
    plt.show(block=False)
    plt.savefig(path_output+"PPA_days_" + str(year) + ".pdf", dpi=600, bbox_inches='tight')
    plt.close()

    # PPA_mask, PPA_mask_xr
    # PPA_mask_labels, PPA_mask_labels_xr
    # Z500_anom, Z500_anom_mask (these are already xarrays)
    # 

    ################################################################################
    # save netcdfs
    PPA_days.to_netcdf(path_output + "PPA_days_" + str(year) + ".nc")
    PPA_mask_filtered_xr.to_netcdf(path_output + "PPA_mask_" + str(year) + ".nc")
    PPA_mask_labels_xr.to_netcdf(path_output + "PPA_mask_with_labels_" + str(year) + ".nc")
    Z500_anom_filtered.to_netcdf(path_output + "Z500_anom_" + str(year) + ".nc")
    Z500_anom_filtered_masked.to_netcdf(path_output + "Z500_anom_mask_" + str(year) + ".nc")


################################################################################
# Plots - REDONE NICER IN PPA_maps.py file... 

# 2023 relative to climatology

years_clim = np.arange(1991, 2020+1)
files = []
for year in years_clim:
    files += glob.glob("{path}/PPA_days_{year}*.nc".format(path=path_output, year=year))


from datetime import datetime
def add_time_dim(xda):
    xda = xda.expand_dims(time = [datetime.now()])
    return xda

PPA_days_all = xr.open_mfdataset(files, preprocess = add_time_dim)
PPA_days_mean = PPA_days_all.mean(dim="time")




fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# generate a basemap with country borders, oceans and coastlines
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='dotted')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
color='gray', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
#cmap = plt.cm.get_cmap('Spectral_r', 8)    # 11 discrete colors
PPA_days_mean.z.plot(transform=ccrs.PlateCarree(), cmap="Spectral_r", cbar_kwargs={'label': "days"})
plt.title("Mean number PPA days for 1991-2020 (April-October)")
plt.show(block=False)
plt.savefig(path_output+"PPA_days_mean.png", dpi=600, bbox_inches='tight')

# 2023 anomalies

PPA_days_2023_anom = PPA_days - PPA_days_mean

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# generate a basemap with country borders, oceans and coastlines
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='dotted')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
color='gray', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
#cmap = plt.cm.get_cmap('Spectral_r', 8)    # 11 discrete colors
PPA_days_2023_anom.z.plot(transform=ccrs.PlateCarree(), cmap="RdBu_r", cbar_kwargs={'label': "days"})
plt.title("Anomaly of PPA days for 2023 compared with 1991-2020 period (April-October)")
plt.show(block=False)
plt.savefig(path_output+"PPA_days_2023_anom.png", dpi=600, bbox_inches='tight')


# original 2023 plot    
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# generate a basemap with country borders, oceans and coastlines
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='dotted')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
color='gray', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
#cmap = plt.cm.get_cmap('Spectral_r', 8)    # 11 discrete colors
PPA_days.z.plot(transform=ccrs.PlateCarree(), cmap="Spectral_r", cbar_kwargs={'label': "days"})
plt.title("Number PPA days for " + str(year) + " (April-October)")
plt.show(block=False)
plt.savefig(path_output+"PPA_days_" + str(year) + ".png", dpi=600, bbox_inches='tight')
