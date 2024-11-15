import geopandas as gpd
import pandas as pd
import datetime as dt
from shapely.geometry import MultiPolygon, Polygon
import matplotlib.pyplot as plt

ids = ['2023_348', '2023_345', '2023_203', '2023_207', '2023_854', '2023_834']
nbac = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp')

# Convert to EPSG 4326 as this is what we will pull FIRMS data with
nbac = nbac.to_crs('EPSG:4326')

# Filter so we're only looking at the fires of interest (the copy is so we're not altering the original dataframe)
nbac_filtered = nbac[nbac['GID'].isin(ids)].copy()

# Add a new column that is a bounding box for each fire
# This will be used to pull FIRMS data just for the fire of interest
nbac_filtered['bbox_coords'] = nbac_filtered['geometry'].apply(lambda geom: geom.bounds)

# Set up the FIRMS API
# go here to get your own: https://firms.modaps.eosdis.nasa.gov/api/map_key
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'
MAP_KEY = FIRMS_map_key
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
try:
    df = pd.read_json(url, typ='series')
except:
    # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
    print("There is an issue with the query. \nTry in your browser: %s" % url)

for idx, fire in nbac_filtered.iterrows():

    # no days goes backwards from the last day
    end_date = dt.datetime.strptime(fire['HS_EDATE'], "%Y/%m/%d").date()
    start_date = dt.datetime.strptime(fire['HS_SDATE'], "%Y/%m/%d").date()
    delta = end_date - start_date
    #no_days = delta.days
    # TODO: Edit this as the maximum no days is 10, so need to concat dfs for the full date range
    # remember it's going backwards
    no_days = 10

    # date to pull, and those before that date defined by no_days
    date = end_date

    # define bounding box of area of interest

    coords = (str(fire['bbox_coords'][0]) + ',' + str(fire['bbox_coords'][1]) + ','
              + str(fire['bbox_coords'][2]) + ',' + str(fire['bbox_coords'][3]))

    #coords = (str(fire['bbox_coords'][2]) + ',' + str(fire['bbox_coords'][1]) + ','
    #          + str(fire['bbox_coords'][0]) + ',' + str(fire['bbox_coords'][3]))

    ## VIIRS_S-NPP_SP
    # Create the corresponding URL for the fire
    area_url_SNPP = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_SNPP_SP/' +
                     str(coords) + '/' + str(no_days) + '/' + str(date))

    # Read the hotspots as a pandas dataframe
    df_SNPP = pd.read_csv(area_url_SNPP)

    # Convert to a geopandas dataframe and make sure we're using the same crs as the bounding boxes
    gdf_SNPP = gpd.GeoDataFrame(
        df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
    )

    ## MODIS_SP
    area_url_MODIS = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/MODIS_SP/' +
                     str(coords) + '/' + str(no_days) + '/' + str(date))

    # Read the hotspots as a pandas dataframe
    df_MODIS = pd.read_csv(area_url_MODIS)

    # Convert to a geopandas dataframe and make sure we're using the same crs as the bounding boxes
    gdf_MODIS = gpd.GeoDataFrame(
        df_MODIS, geometry=gpd.points_from_xy(df_MODIS.longitude, df_MODIS.latitude), crs="EPSG:4326"
    )

    print('test')
    # Double checking this is in the right region
    map = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp')
    ca_map = map[map['iso_a2'] == 'CA']
    ca_map_proj = ca_map.to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=(10, 8))
    ca_map_proj.plot(ax=ax, edgecolor='white', linewidth=1, color='Black')
    gdf_SNPP.plot(ax=ax)
    plt.savefig('test.png')

    print('test')
