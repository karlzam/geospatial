# VIIRS is not the only EO source that provides active fire datasets. Geostationary Operational Environmental
# Satellite (GOES), Moderate Resolution Imaging Spectroradiometer (MODIS) and Landsat are additional EO
# satellites that also have operational active fire monitoring algorithms developed for detecting and
# monitoring wildfires. These different EO sources of hotspot data vary in quality, reliability, and s
# uitability for use in an operation workflow, especially in northern latitudes. We employ a Bayesian
# Neural Network to classify observations as true or false positives, including observations collected
# by other Earth Observations close in time. The uncertainty values associated with the classification
# can be analyzed to determine source-specific trends that could be then used as expert knowledge for f
# uture Bayesian modeling. There will be a dedicated Jupyter notebook resulting from this section.

# This version of script:
# Step One: Pick fires of interest
# Step Two: Load hotspots for GOES, VIIRS, Landsat, and MODIS for entire fire
# Step Three: Flag as inside or outside of respective buffer distances

# Problems with this version of the script:
# - What's the question we're trying to answer here with multiple sources? If the VIIRS hotspots are correct
# considering if there's other hotspots close by?


import geopandas as gpd
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
import matplotlib.patches as mpatches
import utm
from pyproj import CRS
import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union
import math

# Helper function to create a GeoDataFrame
def create_gdf(geometry, crs):
    return gpd.GeoDataFrame([fire], geometry=[geometry], crs=crs)


# Helper function to project and buffer geometries
def project_and_buffer(gdf, target_epsg, buffer_dist):
    gdf_projected = gdf.to_crs(target_epsg)
    gdf_projected['geometry'] = gdf_projected['geometry'].buffer(buffer_dist)
    gdf_projected = gdf_projected.to_crs(epsg=4326)
    return gdf_projected


# Function to add an 'out_buff' column to a GeoDataFrame
def add_outside_buffer_columns(gdf, buffer_gdf, sensor_name):
    """
    Adds a column to the GeoDataFrame indicating if geometries are outside the buffer.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame of points to check.
        buffer_gdf (GeoDataFrame): GeoDataFrame of buffer geometries.
        sensor_name (str): Name of the sensor for logging or debugging.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with 'out_buff' column.
    """
    gdf['out_buff'] = ~gdf['geometry'].apply(lambda geom: buffer_gdf.contains(geom))

    # reproject both buffer and gdf to target epsg for distance calculation
    gdf_projected = gdf.to_crs(target_epsg)
    buffer_gdf_proj = buffer_gdf.to_crs(target_epsg)

    # Also add distance to outside perimeter
    perimeter = buffer_gdf_proj['geometry'][0]
    #gdf['dist_buf'] = perimeter.distance(gdf['geometry'])
    gdf_projected['dist_buf'] = gdf_projected['geometry'].apply(
        lambda point: (point.distance(perimeter.exterior))*(-1) if perimeter.contains(point) else point.distance(perimeter)
    )
    gdf_projected = gdf_projected.to_crs(epsg=4326)

    print(f"Processed {sensor_name} data for outside buffer.")
    return gdf_projected


if __name__ == "__main__":

    # FIRMS API setup
    MAP_KEY = 'e865c77bb60984ab516517cd4cdadea0'
    url = f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}'

    # Fires of interest (selected in QGIS)
    # fire_ids = ['2023_203', '2023_207', '2023_854', '2023_834', '2023_897', '2023_858']
    fire_ids = ['2023_203']

    # Load 2023 NBAC shapefile
    nbac_shapefile = r'C:\Users\kzammit\Documents\shp\nbac\nbac_2023_20240530.shp'
    nbac = gpd.read_file(nbac_shapefile)

    # Convert the CRS to EPSG:4326 for compatibility with FIRMS data
    nbac = nbac.to_crs('EPSG:4326')

    # Filter for the fires of interest
    nbac_filtered = nbac[nbac['GID'].isin(fire_ids)].copy()
    nbac_filtered = nbac_filtered.reset_index()

    # Determine utm zone for each fire (for buffering and projection)
    centroids = nbac_filtered['geometry'].centroid

    # Get lat and long
    lats = centroids.y
    lons = centroids.x

    # Define the UTM zone based on the centroids
    utm_zones = [utm.from_latlon(lat, lon)[2] for lat, lon in zip(lats, lons)]

    # Create a new CRS for each UTM zone
    crs_list = [CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': False}) for zone in utm_zones]

    # Loop over each fire in the filtered dataset
    all_data = gpd.GeoDataFrame()

    for idx, fire in nbac_filtered.iterrows():

        print('Working on fire id ' + str(fire['NFIREID']))

        # Get the appropriate UTM CRS for this fire based on its centroid
        crs = crs_list[idx]

        # Create GeoDataFrames for the original and union geometries
        fire_gdf = create_gdf(fire['geometry'], nbac_filtered.crs)
        fire_union_gdf = create_gdf(unary_union(fire['geometry']).convex_hull, nbac_filtered.crs)

        # Define the target CRS and buffer distances
        target_epsg = crs.to_epsg()
        buffer_distances = {
            "viirs": math.sqrt(2)*375,
            "modis": math.sqrt(2)*1000,
            "landsat": math.sqrt(2)*30,
            "goes": math.sqrt(2)*4000
        }

        # Project and buffer for each dataset
        fire_buffers = {}
        fire_union_buffers = {}

        for sensor, buffer_dist in buffer_distances.items():
            fire_buffers[sensor] = project_and_buffer(fire_gdf, target_epsg, buffer_dist)
            fire_union_buffers[sensor] = project_and_buffer(fire_union_gdf, target_epsg, buffer_dist)

        # Convert fire start and end dates to datetime objects
        end_date = dt.datetime.strptime(fire['HS_EDATE'], "%Y/%m/%d").date()
        start_date = dt.datetime.strptime(fire['HS_SDATE'], "%Y/%m/%d").date()
        delta = end_date - start_date

        # Calculate the number of 10-day intervals
        num_intervals = delta.days // 10

        # Generate a list of dates for each 10-day interval
        date_list = [end_date - dt.timedelta(days=10 * i) for i in range(num_intervals + 1)]
        date_list.reverse()  # To start with the most recent date

        # Get the bounding box coordinates for the current fire using the largest perimeter
        # Add a column with bounding box coordinates for each fire (based on the outside of the nbac)
        bbox_coords = fire_union_buffers['goes']['geometry'].bounds
        bbox_coords = bbox_coords.reset_index(drop=True)
        coords_str = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"

        # define new all dataframes (before it was concatenating too many times)
        df_SNPP_all = gpd.GeoDataFrame()
        df_MODIS_all = gpd.GeoDataFrame()
        df_GOES_all = gpd.GeoDataFrame()
        df_LS_all = gpd.GeoDataFrame()

        # Loop over the date list to download hotspot data for each 10-day period
        for i, date in enumerate(date_list):

            # Determine the number of days for the current period (last period might be less than 10 days)
            period_days = 10 if i < len(date_list) - 1 else delta.days % 10

            # Construct the FIRMS API URLs for VIIRS and MODIS
            url_SNPP = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{coords_str}/{period_days}/{date}'
            url_MODIS = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/MODIS_SP/{coords_str}/{period_days}/{date}'
            url_GOES = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/GOES_NRT/{coords_str}/{period_days}/{date}'
            url_LS = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/LANDSAT_NRT/{coords_str}/{period_days}/{date}'

            # Read the hotspot data from the API into pandas dataframes
            df_SNPP = pd.read_csv(url_SNPP)
            df_MODIS = pd.read_csv(url_MODIS)
            df_GOES = pd.read_csv(url_GOES)
            df_LS = pd.read_csv(url_LS)

            # Filter out columns that are entirely NaN or empty before concatenating
            df_SNPP = df_SNPP.dropna(axis=1, how='all')
            df_MODIS = df_MODIS.dropna(axis=1, how='all')
            df_GOES = df_GOES.dropna(axis=1, how='all')
            df_LS = df_LS.dropna(axis=1, how='all')

            # Concatenate the new data to the existing dataframe
            df_SNPP_all = pd.concat([df_SNPP_all, df_SNPP], ignore_index=True)
            df_MODIS_all = pd.concat([df_MODIS_all, df_MODIS], ignore_index=True)
            df_GOES_all = pd.concat([df_GOES_all, df_GOES], ignore_index=True)
            df_LS_all = pd.concat([df_LS_all, df_LS], ignore_index=True)

        # Convert the SNPP and MODIS dataframes to GeoDataFrames
        gdf_SNPP = gpd.GeoDataFrame(df_SNPP_all, geometry=gpd.points_from_xy(df_SNPP_all.longitude, df_SNPP_all.latitude), crs="EPSG:4326")
        gdf_MODIS = gpd.GeoDataFrame(df_MODIS_all, geometry=gpd.points_from_xy(df_MODIS_all.longitude, df_MODIS_all.latitude), crs="EPSG:4326")
        gdf_GOES = gpd.GeoDataFrame(df_GOES_all, geometry=gpd.points_from_xy(df_GOES_all.longitude, df_GOES_all.latitude), crs="EPSG:4326")
        # satellite and instrument are missing for GOES, so add in some constants in these columns before appending
        gdf_GOES['satellite'] = 'GOES'
        gdf_GOES['instrument'] = 'ABI'
        gdf_LS = gpd.GeoDataFrame(df_LS_all, geometry=gpd.points_from_xy(df_LS_all.longitude, df_LS_all.latitude), crs="EPSG:4326")
        # instrument for landsat is nan
        gdf_LS['instrument'] = 'OLI'

        # Add the 'out_buff' column to each GeoDataFrame
        sensor_buffers = {
            "landsat": fire_union_buffers["landsat"],
            "modis": fire_union_buffers["modis"],
            "viirs": fire_union_buffers["viirs"],
            "goes": fire_union_buffers["goes"]
        }

        # This adds T or F flag if hotspot is within a buffer, and calculates distance to the buffer
        # (negative if within buffer, positive if outside)
        sensor_gdfs = {
            "landsat": add_outside_buffer_columns(gdf_LS, sensor_buffers["landsat"], "Landsat"),
            "modis": add_outside_buffer_columns(gdf_MODIS, sensor_buffers["modis"], "MODIS"),
            "viirs": add_outside_buffer_columns(gdf_SNPP, sensor_buffers["viirs"], "SNPP"),
            "goes": add_outside_buffer_columns(gdf_GOES, sensor_buffers["goes"], "GOES")
        }

        # TODO: Combine all datasets? Adding source type (0, 1, 2, 3) column
        # To do so, must understand the differences and similarities between datasets



        print('test')






