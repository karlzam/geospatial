'''
Calculate monthly climatology files for Z500. 
For use with PPA identification code. 

Author: Piyush Jain
piyush.jain@nrcan-rncan.gc.ca
jain@ualberta.ca

23 Sept 2024
'''


# import statements
import xarray as xr
import numpy as np
import glob
import re
#conda install dask --force-reinstall --solver=libmamba


# set values
years = np.arange(1991, 2020+1) #1991 - 2020 for climatology
months = np.arange(1, 12+1) # test data monthly
path_data = "/Users/piyush/RESEARCH/PROJECTS/Z500 Forecasting analysis/Data/Z500_ERA5/geopotential/"
path_data = "/Users/piyush/RESEARCH/PROJECTS/Synoptic Analysis/DATA_for_clustering/geopotential/"
area = "NA"

'''
FIRST VERSION:
uses single day of each year to form climatology
'''

# loop through months and years and calculate climatology
for month in months:

    print(month)

    # load first year
    #Z500 = iris.load_cube(path_data + "geopotential_500_NA_" + str(years[0]) + "_" + str(month) + "_daily.mean.nc")
    Z500  = xr.open_dataset(path_data + "geopotential_500_" + area + "_" + str(years[0]) + "_" + str(month) + "_daily.mean.nc", decode_times=True)

    # fill any masked values (there should be none anyway)
    Z500_clim = Z500.copy(deep=True)
    Z500sq_clim = np.square(Z500.copy(deep=True))
    #Z500_clim.z.values = Z500_clim.z.values.filled(0.)


    # loop through remaining years
    for year in years[1:]:
        Z500  = xr.open_dataset(path_data + "geopotential_500_" + area + "_" + str(year) + "_" + str(month) + "_daily.mean.nc", decode_times=True)
        # drop leap day
        Z500 = Z500.sel(time=~((Z500.time.dt.month == 2) & (Z500.time.dt.day == 29)))
        Z500_clim.z.values += Z500.z.values
        Z500sq_clim.z.values += np.square(Z500.z.values)

    # divide by number of years for mean
    Z500_clim.z.values /= len(years)
    Z500sq_clim.z.values /= len(years)


    # write out
    #iris.save(Z500_clim, path_data + "geopotential_500_NA_climatology_" + str(month) + "_daily.mean.nc")
    Z500_clim.to_netcdf(path_data + "climatology_daily/geopotential_500_" + area + "_climatology_" + str(month) + "_daily.mean.nc")
    Z500sq_clim.to_netcdf(path_data + "climatology_daily/geopotential_500_squared_" + area + "_climatology_" + str(month) + "_daily.mean.nc")

################################################################################
# save as single file
del Z500_clim, Z500sq_clim

# https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Z500
paths = glob.glob(path_data + "climatology_daily/geopotential_500_" + area + "_climatology_*_daily.mean.nc")
paths = natural_sort(paths)
Z500_clim= xr.open_mfdataset(paths, concat_dim='time', combine='nested')
Z500_clim.to_netcdf(path_data + "climatology_daily/geopotential_500_" + area + "_climatology_daily.mean.nc")

# Z500sq
paths = glob.glob(path_data + "climatology_daily/geopotential_500_squared_" + area + "_climatology_*_daily.mean.nc")
paths = natural_sort(paths)
Z500sq_clim= xr.open_mfdataset(paths, concat_dim='time', combine='nested')
Z500sq_clim.to_netcdf(path_data + "climatology_daily/geopotential_500_squared_" + area + "_climatology_daily.mean.nc")

# calculate variance and standard deviation and output
Var_Z500 = Z500sq_clim - np.square(Z500_clim)
sigma_Z500 = np.sqrt(Var_Z500)
sigma_Z500.to_netcdf(path_data + "climatology_daily/geopotential_500_sigma_" + area + "_climatology_daily.mean.nc")

'''
# maximum raw values in the time series (optional for heat dome paper)
Z500  = xr.open_dataset(path_data + "geopotential_500_NA_" + str(2021) + "_" + str(6) + "_daily.mean.nc", decode_times=True)
maxVal = np.max(Z500.z.values.flatten())/9.80665
print(maxVal)
'''

'''
SECOND VERSION:
uses 4 week moving window to form climatology as per Miller et al. 2020
This is the version we use in the PPA algorithm
'''

# load first year
Z500 = xr.open_mfdataset(path_data + "geopotential_500_NA_"+str(years[0])+"_*_daily.mean.nc", decode_times=True)
Z500_clim_4wk = Z500.copy().rolling(time=29, center=True).sum()
Z500sq_clim_4wk = np.square(Z500).rolling(time=29, center=True).sum()

# loop through remaining years
for year in years[1:]:
    Z500 = xr.open_mfdataset(path_data + "geopotential_500_NA_"+str(year)+"_*_daily.mean.nc", decode_times=True)
    # drop leap day
    Z500 = Z500.sel(time=~((Z500.time.dt.month == 2) & (Z500.time.dt.day == 29)))
    Z500_clim_4wk.z.values += Z500.rolling(time=29, center=True).sum().z.values
    Z500sq_clim_4wk.z.values += np.square(Z500).rolling(time=29, center=True).sum().z.values

# divide by number of years for mean
Z500_clim_4wk.z.values /= (29*len(years))
Z500sq_clim_4wk.z.values /= (29*len(years))

# write out
Z500_clim_4wk.to_netcdf(path_data + "climatology_daily_4wk/geopotential_500_" + area + "_climatology_daily_4wk.mean.nc")
Z500sq_clim_4wk.to_netcdf(path_data + "climatology_daily_4wk/geopotential_500_squared_" + area + "_climatology_daily_4wk.mean.nc")

################################################################################

# calculate variance and standard deviation and output

Var_Z500_4wk = Z500sq_clim_4wk - np.square(Z500_clim_4wk)
sigma_Z500_4wk = np.sqrt(Var_Z500_4wk)
sigma_Z500_4wk.to_netcdf(path_data + "climatology_daily_4wk/geopotential_500_sigma_" + area + "_climatology_daily_4wk.mean.nc")

################################################################################
# DO NOT NEED TO RUN PAST HERE
# some tests to make sure these are both working okay
import matplotlib.pyplot as plt


fig = plt.figure()
Z500_clim.z[200,:,:].plot()
plt.show(block=False)

fig = plt.figure()
Z500_clim_4wk.z[200,:,:].plot()
plt.show(block=False)

fig = plt.figure()
Z500sq_clim.z[200,:,:].plot()
plt.show(block=False)

fig = plt.figure()
Z500sq_clim_4wk.z[200,:,:].plot()
plt.show(block=False)

fig = plt.figure()
sigma_Z500.z[200,:,:].plot()
plt.show(block=False)

# THIS PPEARS TO BE 29 TIMES HIGHER THAN IT SHOULD BE
fig = plt.figure()
sigma_Z500_4wk.z[200,:,:].plot()
plt.show(block=False)

