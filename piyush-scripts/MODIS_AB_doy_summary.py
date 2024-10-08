# calculate area burned for North America using MODIS AB product and Google
# Earth Engine.
# Final version
# Piyush Jain,

'''
Conda environment:

conda create -n gee python=3.10
conda activate gee
conda install -c conda-forge earthengine-api
conda install ipython # optional
conda install numpy
conda install -c conda-forge cartopy
conda install pandas

pip install cartoee # this doesn't seem to work
conda install -c conda-forge geemap # this isn't really useful either because interactive

# authenticate google earth engine
earthengine authenticate
'''

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

# Initialize the Earth Engine module.
ee.Authenticate()
ee.Initialize(project="karlzam")

path = "/Users/kzammit/Documents/VIIRS-data-GEE/"

################################################################################
# define areas of area of interest

# general bounding box
area_of_interest = ee.Geometry.Rectangle([-173, 24, -52, 72], 'EPSG:4326', False)

# North America (This is the large scale international boundary polygons)
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.inList("wld_rgn", ['North America'])).filter(ee.Filter.inList("country_na", ['Canada', 'United States', 'United States (Alaska)']))

# create image mask for north america (this is faster for subsetting than Using
# original featurecollection "countries")
# Mask that creates pixel values of 0 everywhere except for in Canada
countries_mask = countries.map(
    lambda feature: feature.set('n', 1)
).reduceToImage(
    properties = ['n'],
    reducer = ee.Reducer.first()
).mask()

# Do the same thing for Canada and the USA
canada = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.inList("wld_rgn", ['North America'])).filter(ee.Filter.inList("country_na", ['Canada']))
unitedstates = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.inList("wld_rgn", ['North America'])).filter(ee.Filter.inList("country_na", ['United States', 'United States (Alaska)']))

canada_mask = canada.map(
    lambda feature: feature.set('n', 1)
).reduceToImage(
    properties = ['n'],
    reducer = ee.Reducer.first()
).mask()

unitedstates_mask = unitedstates.map(
    lambda feature: feature.set('n', 1)
).reduceToImage(
    properties = ['n'],
    reducer = ee.Reducer.first()
).mask()

################################################################################

# Create a dataframe of days of year (not sure why... this looks like [0 1], [1 2] etc
doyList = ee.List.sequence(1, 366)
print(doyList.getInfo())
df_final = pd.DataFrame(data = doyList.getInfo(), columns=["doy"])

# Define the years for analysis
years = np.arange(2001, 2002+1)
#years = np.arange(2001, 2021+1)

# Set the region for analysis
# Created masks for NA, Canada, USA
region = "North_America" # NorthAmerica, Canada, United_States

# Set the correct mask
region_mask = countries_mask
if region == "Canada":
    region_mask = canada_mask
elif region == "United_States":
    region_mask = unitedstates_mask

################################################################################
# For each year defined above
for year in years:

    print("Processing year " + str(year))

    # date range
    date_range = {
      'start': str(year)+'-01-01',
      'end': str(year)+'-12-31',
    }

    # load data etc.
    # This is the MODIS burned area product
    # An image collection of all images from the start date to the end date defined, in the specified region of interest
    collection = ee.ImageCollection("MODIS/006/MCD64A1").filterDate(date_range['start'], date_range['end']).filterBounds(area_of_interest)

    # toBands() converts a collection to a single multi-band image containing all bands of every image in that collection
    # to determine what bands a collection has: print('All band names', collection.first().bandNames().getInfo())
    burnDate = collection.select('BurnDate').toBands()#.reproject(crs=...)

    # EPSG:4326
    # GEE says print(proj) is supposed to return the projection like "EPSG"
    # Image.select() = select a specific band from the image
    proj = burnDate.select(0).projection()

    # Determine the native resolution of the image
    # Each band of an image can have a different scale and/or projection so that's why you do it on one band
    scale = proj.nominalScale()

    # mask to NA
    burnDate = burnDate.mask(region_mask)

    # define pixel area layer
    # Image.pixelArea() generates an image in which the value of each pixel is the area of that pixel in square metres
    # Returned image has a single band called "area"
    # And then reproject (force an image to be computed in a given projection and resolution)
    # This is calculating the area of the "area of interest" which was defined at the beginning of the script
    area = ee.Image.pixelArea().reproject(proj, None, scale).clip(area_of_interest)

    ################################################################################

    # reduceRegion is used to summarize pixel values over a specified region and allows you to perform statistical operations within the area of interest
    # so this is getting a histogram of occurances for each date,
    # calculating a frequency histogram on the burn date which is a compressed multi-band image from the feature collection of MODIS burned areas
    # this is getting monthly results within the area
    dic1 = burnDate.reduceRegion(
      reducer = ee.Reducer.frequencyHistogram(),
      geometry =  area_of_interest,
      scale = scale,
      maxPixels = 1e13)

    #
    dic2 = dic1.getInfo()

    # merge dictionaries
    # merge monthly dictionaries to be by day
    result = {}
    for k in dic2.keys():
        result.update(dic2[k])

    # set items up into a dataframe
    df = pd.DataFrame(list(result.items()),columns = ['doy','sum_burned_pixels'])

    # change type of doy to ints
    df[["doy"]] = df[["doy"]].astype(int)

    # sort the dataframe so it's in order of doy
    df = df.sort_values(by=['doy'])
    res = scale.getInfo()

    # Converting into hectares - what was it in before?
    # Res is 463.31m (native data resolution)
    # m to ha is /1e4
    df["AB_"+str(year)] = df[["sum_burned_pixels"]]*res**2/1e4 #ha

    # Create a final dataframe that's AB and year by day (I think it's area burned)
    df_final = pd.merge(df_final, df[["doy", "AB_"+str(year)]], on="doy", how="left").fillna({'AB':0})

################################################################################
# process data

# save data
df_final.to_csv(path+"MODIS_AB_"+region+"_"+str(years[0])+"_"+str(years[-1])+".csv")

#import datetime
#dates = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%d-%m-%Y")

# get three biggest years
totalab = df_final.sum(axis=0)[1:].sort_values(ascending=False)
top3 = totalab[0:3]
ntraj=len(years)
colors = plt.cm.jet(np.linspace(0,1,ntraj))# Initialize holder for trajectories

# This plots the lineplot of cumulative burned area over years
#years = years[:-1] # chop off 2023
n = len(years[1:])
#colors = plt.cm.jet(np.linspace(0,1,n))
colors = plt.get_cmap("OrRd")(np.linspace(0,1,n))
fig, ax = plt.subplots(figsize=(9,5))
for i, year in enumerate(years[1:]):
    col = colors[i]
    #col = "grey"
    #if year == years[-1]:
    #    col = "darkred"
    #if year == 2016:
    #    col = "orange"
    # Calculating cumulative burned area per day for the year
    cumAB = np.nancumsum(df_final["AB_"+str(year)])
    print(str(year) + " AB: " + str(cumAB[-1]))
    plt.plot(df_final.doy, cumAB/1e6, color=col, linewidth=1)
#plt.axvline(x = 181, color="green")
#plt.axvline(x = 212, color="green")
plt.xlabel("day of year")
plt.ylabel(region.replace("_", " ")+" Area Burned (Mha)")
plt.gca().set_xlim(-5, 400)
#ticks = plt.gca().get_xticklabels()
#ticks = ticks[0:(len(ticks)-1)]
#plt.gca().set_xticklabels(ticks)
for i, val in enumerate(top3):
    plt.text(367, (val-1e5)/1e6, str(top3.index[i]).replace("AB_", ""), fontsize=2, fontweight="bold", color="#666666")
plt.setp(plt.gca().get_xticklabels()[-1], visible=False)
plt.show(block=False)
plt.savefig(path+region+"_Area_Burned_doy.pdf")

###

# monthly summaries
df_final_monthly = pd.DataFrame(data=np.arange(1,13), columns=["month"])

for year in years[1:]:
    df = df_final[["doy", "AB_"+str(year)]]
    dates = [datetime.datetime(year, 1, 1) + datetime.timedelta(int(doy) - 1) for doy in df.doy.values]
    df["date"] = pd.to_datetime(dates)
    dg = df.groupby(pd.Grouper(key='date', freq='1M')).sum() # groupby each 1 month
    dg.index = dg.index.strftime('%B')
    df_final_monthly["AB_"+str(year)] = dg["AB_"+str(year)].values[0:12]

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(9*1.2,5*1.2))
plt.imshow( df_final_monthly.iloc[:,1:]/1e6, interpolation='nearest',
    cmap=plt.get_cmap("Reds"))
ax.set_yticks(range(12))
ax.set_yticklabels(dg.index.values[0:12])
ax.set_xticks(range(len(years[1:])))
ax.set_xticklabels(years[1:], rotation=90)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
cb = plt.colorbar(cax=cax)
cb.ax.set_title('Mha')

plt.show(block=False)
plt.savefig(path+region+"_Area_Burned_month.pdf")


#plt.figure()
#plt.plot(df_final.sum(axis=0)[2:], '.-')
#plt.show(block=False)
