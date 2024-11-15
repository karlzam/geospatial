import geopandas as gpd
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

# Define the fire IDs of interest
fire_ids = ['2023_348', '2023_345', '2023_203', '2023_207', '2023_854', '2023_834']

# Load the shapefile containing fire data
nbac_shapefile = r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp'
nbac = gpd.read_file(nbac_shapefile)

# Convert the CRS to EPSG:4326 for compatibility with FIRMS data
nbac = nbac.to_crs('EPSG:4326')

# Filter for the fires of interest
nbac_filtered = nbac[nbac['GID'].isin(fire_ids)].copy()

# Add a column with bounding box coordinates for each fire
nbac_filtered['bbox_coords'] = nbac_filtered['geometry'].apply(lambda geom: geom.bounds)

# FIRMS API setup (replace with your own MAP_KEY)
MAP_KEY = 'e865c77bb60984ab516517cd4cdadea0'
url = f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}'

# Test connection to FIRMS API
try:
    df = pd.read_json(url, typ='series')
except Exception as e:
    print(f"Error connecting to FIRMS API: {e}")
    print(f"Check URL in your browser: {url}")

# Loop over each fire in the filtered dataset
for idx, fire in nbac_filtered.iterrows():
    # Convert fire start and end dates to datetime objects
    end_date = dt.datetime.strptime(fire['HS_EDATE'], "%Y/%m/%d").date()
    start_date = dt.datetime.strptime(fire['HS_SDATE'], "%Y/%m/%d").date()
    delta = end_date - start_date

    # Calculate the number of 10-day intervals
    num_intervals = delta.days // 10

    # Generate a list of dates for each 10-day interval
    date_list = [end_date - dt.timedelta(days=10 * i) for i in range(num_intervals + 1)]
    date_list.reverse()  # To start with the most recent date

    # Get the bounding box coordinates for the current fire
    bbox_coords = fire['bbox_coords']
    coords_str = f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}"

    # Initialize dataframes for storing hotspots data
    df_SNPP_all = pd.DataFrame()
    df_MODIS_all = pd.DataFrame()

    # Loop over the date list to download hotspot data for each 10-day period
    for i, date in enumerate(date_list):
        # Determine the number of days for the current period (last period might be less than 10 days)
        period_days = 10 if i < len(date_list) - 1 else delta.days % 10

        # Construct the FIRMS API URLs for VIIRS and MODIS
        url_SNPP = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{coords_str}/{period_days}/{date}'
        url_MODIS = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/MODIS_SP/{coords_str}/{period_days}/{date}'

        # Read the hotspot data from the API into pandas dataframes
        df_SNPP = pd.read_csv(url_SNPP)
        df_MODIS = pd.read_csv(url_MODIS)

        # Filter out columns that are entirely NaN or empty before concatenating
        df_SNPP = df_SNPP.dropna(axis=1, how='all')
        df_MODIS = df_MODIS.dropna(axis=1, how='all')

        # Concatenate the new data to the existing dataframe
        df_SNPP_all = pd.concat([df_SNPP_all, df_SNPP], ignore_index=True)
        df_MODIS_all = pd.concat([df_MODIS_all, df_MODIS], ignore_index=True)

    # Convert the SNPP and MODIS dataframes to GeoDataFrames
    gdf_SNPP = gpd.GeoDataFrame(df_SNPP_all, geometry=gpd.points_from_xy(df_SNPP_all.longitude, df_SNPP_all.latitude), crs="EPSG:4326")
    gdf_MODIS = gpd.GeoDataFrame(df_MODIS_all, geometry=gpd.points_from_xy(df_MODIS_all.longitude, df_MODIS_all.latitude), crs="EPSG:4326")

    # Load the map of Canada (or other region of interest)
    map_path = r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp'
    map_data = gpd.read_file(map_path)
    ca_map = map_data[map_data['iso_a2'] == 'CA']
    ca_map_proj = ca_map.to_crs(epsg=4326)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 8))
    ca_map_proj.plot(ax=ax, edgecolor='white', linewidth=1, color='black')
    plot_polygon(fire['geometry'], ax=ax, color='white', zorder=1)
    gdf_SNPP.plot(ax=ax, color='red', zorder=2)
    gdf_MODIS.plot(ax=ax, color='orange', zorder=3)

    # Set the plot limits based on the fire's bounding box
    ax.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))

    # Save the plot to a file
    output_path = r'C:\Users\kzammit\Documents\DL-chapter\hotspots-fire-' + str(fire['GID']) + '.png'
    plt.savefig(output_path)

    print(f"Saved plot for fire {fire['GID']} to {output_path}")
