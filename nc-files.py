from matplotlib import pyplot as plt
import pandas as pd
import netCDF4
import xarray as xr

# This one works
#fp=r'C:\Users\kzammit\Documents\FWIq95_1991_2020.nc'

viirs_data = r'C:\Users\kzammit\Documents\VIIRS-data-VNP02IMG\VNP02IMG.A2018209.2018.002.2021095180702.nc'

GOES_data = r'C:\Users\kzammit\Documents\GOES\noaa-goes16\ABI-L1b-RadC\2019\300\00\OR_ABI-L1b-RadC-M6C01_G16_s20193000001341_e20193000004115_c20193000004197.nc'

# These are weird - from download script
# https://lpdaac.usgs.gov/products/vnp14imgv002/
#fp = r'C:\Users\kzammit\Documents\VIIRS-data-VNP14IMG\2018\207\VNP14IMG.A2018207.2100.002.2024081023159.nc'

# Downloaded this directly from earth data
fp = r'C:\Users\kzammit\Documents\VIIRS-data-VNP14IMG\Direct\VNP14IMG.A2018207.2100.001.2018208043638.nc'

#nc = netCDF4.Dataset(fp)

ds = xr.open_dataset(GOES_data)

#ds['fire mask'].plot()
#plt.savefig('test.png')

print('test')

plt.imshow(ds['fire mask'])
plt.show()

print('test')