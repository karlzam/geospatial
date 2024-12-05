# Question two: can we classify VIIRS hotspots as T or F positive using a BNN?
# " we present a method for training a probabilistic BNN to perform classification on the vectorized fire data into
# the classes: true or false positive"
import math

# Step One: Pull all VIIRS hotspots for Canada for a specific day
# Step Two: Load NBAC, NFDB, and persistent hot spots
# Step Three: Buffer these by 375*sqrt(2)
# Step Four: Flag as outside or inside a known boundary and set this is as "0" and "1" for TP/FP
# Step Five: Apply to hotspots from a different day and classify as TP/FP

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import shapefile as shp
import numpy as np
from shapely.geometry import box
import math


###### User Inputs ######

# Obtained from: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/
nbac_shp = r'C:\Users\kzammit\Documents\shp\nbac\nbac_2023_20240530.shp'

# At the time of writing this script, NFDB polygons were not available for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt
nfdb_shp = r'C:\Users\kzammit\Documents\shp\nfdb\NFDB_point_20240613.shp'

# Obtained from Piyush
pers_hs_shp = r'C:\Users\kzammit\Documents\shp\pers-hs\m3mask5_lcc.shp'

# Obtained from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/
# ne_10m_admin_0_countries.zip
nat_earth_shp = r'C:\Users\kzammit\Documents\shp\nat-earth\ne_10m_admin_0_countries.shp'

# Date to focus analysis on (training used 2023, val used 2022)
doi = '2023/09/23'
yoi = 2023

# FIRMS may key to use API
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'
doi_firms = '2023-09-23'

# Buffer distance
buffer_dist = 375*math.sqrt(2)


###### Code ######

### NBAC
# Import all fires for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/
nbac = gpd.read_file(nbac_shp)

# NBAC has agency start date as well as hotspot start date, and these often do not align
# If one of them is missing, the fill value is "0000/00/00"
# Create start and end date columns from the earlier or later of the two respectively
nbac['start_date'] = 'tbd'

# If either agency or hotspot start date is empty, assign the other as the start date 
nbac.loc[nbac['HS_SDATE'] == '0000/00/00', 'start_date'] = nbac['AG_SDATE']
nbac.loc[nbac['AG_SDATE'] == '0000/00/00', 'start_date'] = nbac['HS_SDATE']

# Pick the earlier of the two for the start date, excluding the empties 
nbac.loc[(nbac['HS_SDATE'] <= nbac['AG_SDATE']) & (nbac['HS_SDATE'] != '0000/00/00'), 'start_date'] = nbac['HS_SDATE']
nbac.loc[(nbac['AG_SDATE'] <= nbac['HS_SDATE']) & (nbac['AG_SDATE'] != '0000/00/00'), 'start_date'] = nbac['AG_SDATE']

# Do the same steps for the end date
nbac['end_date'] = 'tbd'
nbac.loc[nbac['HS_EDATE'] == '0000/00/00', 'end_date'] = nbac['AG_EDATE']
nbac.loc[nbac['AG_EDATE'] == '0000/00/00', 'end_date'] = nbac['HS_EDATE']
nbac.loc[(nbac['HS_EDATE'] >= nbac['AG_EDATE']) & (nbac['HS_EDATE'] != '0000/00/00'), 'end_date'] = nbac['HS_EDATE']
nbac.loc[(nbac['AG_EDATE'] >= nbac['HS_EDATE']) & (nbac['AG_EDATE'] != '0000/00/00'), 'end_date'] = nbac['AG_EDATE']

# There are some cases where there is no agency date OR hotspot date, drop these 
nbac = nbac.drop(nbac[(nbac.start_date == '0000/00/00')].index)
nbac = nbac.drop(nbac[(nbac.end_date == '0000/00/00')].index)

# Filter the hotspots so we're only looking at fires which contain Sept 23 within their date range 
date_format = '%Y/%m/%d'
date_obj = datetime.strptime(doi, date_format)
nbac['doi'] = 0
nbac = nbac.assign(start_dt = lambda x: pd.to_datetime(x['start_date'], format=date_format))
nbac = nbac.assign(end_dt = lambda x: pd.to_datetime(x['end_date'], format=date_format))
nbac.loc[(nbac['start_dt'] <= date_obj) & (nbac['end_dt'] >= date_obj), "doi"] = 1

# Create a new dataframe for only fires containing the date of interest
nbac_doi = nbac[nbac['doi']==1]
nbac_doi = nbac_doi.reset_index()


### NFDB
nfdb = gpd.read_file(nfdb_shp)
nfdb_doi = nfdb[nfdb['YEAR']==yoi]
nfdb_doi = nfdb_doi.drop(nfdb_doi[(nfdb_doi.OUT_DATE == '0000/00/00')].index)
nfdb_doi['doi'] = 0
nfdb_doi = nfdb_doi.assign(start_dt = lambda x: pd.to_datetime(x['REP_DATE'], format=date_format))
nfdb_doi = nfdb_doi.assign(end_dt = lambda x: pd.to_datetime(x['OUT_DATE'], format=date_format))
nfdb_doi.loc[(nfdb_doi['start_dt'] <= date_obj) & (nfdb_doi['end_dt'] >= date_obj), "doi"] = 1
nfdb_doi = nfdb_doi[nfdb_doi['doi']==1]
nfdb_doi = nfdb_doi.reset_index()


### Persistent heat sources
pers_hs = gpd.read_file(pers_hs_shp)
cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
pers_hs_cad = pers_hs[pers_hs['prov'].isin(cad_provs)]


### Apply buffers to all sounds to account for hotspot resolution
# NAD83 (EPSG 3978) is commonly used for Canada
# Helper function to project and buffer geometries
def project_and_buffer(gdf, target_epsg, buffer_dist=buffer_dist):
    gdf_projected = gdf.to_crs(target_epsg)
    gdf_projected['geometry'] = gdf_projected['geometry'].buffer(buffer_dist)
    gdf_projected = gdf_projected.to_crs(epsg=4326)
    return gdf_projected

# Commenting this out because it takes a few minutes to run
#target_epsg = 3978
#nbac_buff = project_and_buffer(nbac_doi, target_epsg)
#nfdb_buff = project_and_buffer(nfdb_doi, target_epsg)
#pers_hs_cad_buff = project_and_buffer(pers_hs_cad, target_epsg)
#nbac_buff.to_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\nbac-buff.shp')
#nfdb_buff.to_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\nfdb-buff.shp')
#pers_hs_cad_buff.to_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\pers-hs-cad-buff.shp')

nbac_buff = gpd.read_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\nbac-buff.shp')
nfdb_buff = gpd.read_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\nfdb-buff.shp')
pers_hs_cad_buff = gpd.read_file(r'C:\Users\kzammit\Documents\DL-chapter\shp\pers-hs-cad-buff.shp')

nbac_doi = nbac_doi.to_crs(epsg=4326)
nfdb_doi = nfdb_doi.to_crs(epsg=4326)

# check the buffering was done correctly
#fig, ax = plt.subplots(figsize=(10,8))
#nbac_buff.plot(ax=ax, color='red', linewidth=1)
#nbac_doi.plot(ax=ax, color='black', linewidth=1)
#plt.savefig('test.png')

# Join them all into a single dataframe (it's ok that it's messy, we'll just be using the geometry column)
# The epsg code are already the same for all (defined in the function)
df_perims = pd.concat([nbac_buff, nfdb_buff, pers_hs_cad_buff])


### Canada
ne = gpd.read_file(nat_earth_shp)
cad = ne[ne['ADMIN']=='Canada']

# Create a bounding box around Canada (this will include some of the States, but we'll fix this later)
bbox_coords = cad.bounds
bbox_coords = bbox_coords.reset_index()
coords = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"


### Hotspots
# Only SNPP and NOAA20 are available for 2023 (NOAA21 came after)

# FIRMS
MAP_KEY = FIRMS_map_key
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
try:
    df = pd.read_json(url, typ='series')
except:
    # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
    print("There is an issue with the query. \nTry in your browser: %s" % url)

## VIIRS_S-NPP
area_url_SNPP = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_SNPP_SP/' +
                 str(coords) + '/' + str(1) + '/' + str(doi_firms))
df_SNPP = pd.read_csv(area_url_SNPP)

gdf_SNPP = gpd.GeoDataFrame(
    df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
)

## VIIRS_NOAA20
area_url_NOAA20 = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/' +
                 str(coords) + '/' + str(1) + '/' + str(doi_firms))
df_NOAA20 = pd.read_csv(area_url_NOAA20)

gdf_NOAA20 = gpd.GeoDataFrame(
    df_NOAA20, geometry=gpd.points_from_xy(df_NOAA20.longitude, df_NOAA20.latitude), crs="EPSG:4326"
)

# Remove hotspots outside of Canada
gdf_SNPP = gpd.sjoin(gdf_SNPP, cad, predicate='within', how='left')
gdf_SNPP = gdf_SNPP[gdf_SNPP.index_right.notnull()].copy()
gdf_SNPP = gdf_SNPP.drop(gdf_SNPP.columns[16:], axis=1)

gdf_NOAA20 = gpd.sjoin(gdf_NOAA20, cad, predicate='within', how='left')
gdf_NOAA20 = gdf_NOAA20[gdf_NOAA20.index_right.notnull()].copy()
gdf_NOAA20= gdf_NOAA20.drop(gdf_NOAA20.columns[15:], axis=1)


### Determine hotspots inside and outside of buffered regions
gdf_SNPP['sat'] = 'SNPP'
gdf_NOAA20['sat'] = 'NOAA20'
df_hotspots = pd.concat([gdf_SNPP, gdf_NOAA20])

# Perform a spatial join to find hotspots within known geometries
# how returns the geometry column for only the specified dataframe
joined = gpd.sjoin(df_hotspots, df_perims, predicate='within', how='left')

joined = joined.drop(joined.columns[19:], axis=1)
joined['Class'] = 1
joined.loc[joined.index_right0.isnull(), 'Class'] = 0
joined = joined.drop(['index', 'index_right0', 'type', 'geometry'], axis=1)

# Balance dataset by randomly dropping positive examples
num_fp = (joined['Class'] == 0).sum()
fp_df = joined[joined['Class'] == 0]

num_tp = (joined['Class'] == 1).sum()
tp_df = joined[joined['Class'] == 1]

if num_tp > num_fp:
    fp_sample = fp_df.sample(n=num_fp, random_state=42)
    tp_sample = tp_df.sample(n=num_fp, random_state=42)

elif num_fp > num_tp:
    fp_sample = fp_df.sample(n=num_tp, random_state=42)
    tp_sample = tp_df.sample(n=num_tp, random_state=42)

df = pd.concat([fp_sample, tp_sample])

df.to_excel(r'C:\Users\kzammit\Documents\DL-chapter\train.xlsx', index=False)






