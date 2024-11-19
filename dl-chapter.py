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

def plot_fires():
    # Load the map of Canada (or other region of interest)
    map_path = r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp'
    map_data = gpd.read_file(map_path)
    ca_map = map_data[map_data['iso_a2'] == 'CA']
    ca_map_proj = ca_map.to_crs(epsg=4326)

    # Plot the data
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))

    # all hotspots overlapping with no buffer
    ca_map_proj.plot(ax=ax1, edgecolor='white', linewidth=1, color='black')
    plot_polygon(fire['geometry'], ax=ax1, color='white', zorder=1)
    gdf_SNPP.plot(ax=ax1, color='red', zorder=2)
    gdf_MODIS.plot(ax=ax1, color='orange', zorder=3)
    gdf_GOES.plot(ax=ax1, color='yellow', zorder=4)
    gdf_LS.plot(ax=ax1, color='blue', zorder=5)
    ax1.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax1.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax1.set_title('All HS, no buffer')

    # VIIRS
    ca_map_proj.plot(ax=ax2, edgecolor='white', linewidth=1, color='black', zorder=1)
    fire_buff_viirs.plot(ax=ax2, zorder=2, color='white')
    gdf_SNPP.plot(ax=ax2, color='red', zorder=3)
    ax2.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax2.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax2.set_title('VIIRS HS, VIIRS buffer')

    # MODIS
    ca_map_proj.plot(ax=ax3, edgecolor='white', linewidth=1, color='black', zorder=1)
    fire_buff_modis.plot(ax=ax3, zorder=2, color='white')
    gdf_MODIS.plot(ax=ax3, color='orange', zorder=3)
    ax3.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax3.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax3.set_title('MODIS HS, MODIS buffer')

    # Landsat
    ca_map_proj.plot(ax=ax4, edgecolor='white', linewidth=1, color='black', zorder=1)
    fire_buff_landsat.plot(ax=ax4, zorder=2, color='white')
    gdf_LS.plot(ax=ax4, color='blue', zorder=3)
    ax4.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax4.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax4.set_title('LandSat HS, LandSat buffer')

    # GOES
    # TODO: Figure out how to do GOES buffer
    ca_map_proj.plot(ax=ax5, edgecolor='white', linewidth=1, color='black', zorder=1)
    plot_polygon(fire['geometry'], ax=ax5, color='white', zorder=2)
    gdf_GOES.plot(ax=ax5, color='yellow', zorder=3)
    ax5.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax5.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax5.set_title('GOES HS, No Buffer')

    NBAC_patch = mpatches.Patch(color='white', label='NBAC-buff')
    SNPP_patch = mpatches.Patch(color='red', label='SNPP')
    MODIS_patch = mpatches.Patch(color='orange', label='MODIS')
    GOES_patch = mpatches.Patch(color='yellow', label='GOES')
    LS_patch = mpatches.Patch(color='blue', label='Landsat')

    fig.suptitle(f"Fire {fire['GID']}")
    fig.delaxes(ax=ax6)
    fig.text(0.5, 0.04, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')

    # fig.legend(handles=[SNPP_patch, MODIS_patch, GOES_patch, LS_patch], loc='lower right', bbox_to_anchor=(1.05, 1))
    fig.legend(handles=[NBAC_patch, SNPP_patch, MODIS_patch, GOES_patch, LS_patch], loc='center', bbox_to_anchor=(0.75, 0.35))

    # Save the plot to a file
    output_path = r'C:\Users\kzammit\Documents\DL-chapter\hotspots-fire-' + str(fire['GID']) + '.png'
    plt.savefig(output_path, bbox_inches='tight')

    print(f"Saved plot for fire {fire['GID']} to {output_path}")


def process_gridding_and_frp(source_gdf, grid_gdf, source_name, frp_available=True):
    """
    Processes gridding and calculates cumulative FRP for a given data source.

    Args:
        source_gdf (GeoDataFrame): The GeoDataFrame containing hotspots data for a specific source.
        grid_gdf (GeoDataFrame): The GeoDataFrame containing the grid cells.
        source_name (str): The name of the data source (e.g., 'goes', 'viirs', 'modis', 'ls').
        frp_available (bool): Flag to indicate if FRP calculation is available for this source.

    Returns:
        GeoDataFrame: The updated grid_gdf with added columns for hotspot count and (optionally) cumulative FRP.
    """
    # Perform spatial join between the source data and the grid
    grid_source = gpd.sjoin(source_gdf, grid_gdf, how='left', predicate='within')

    # Add a column for hotspot counts
    count_col = f'{source_name}_hs'
    grid_source[count_col] = 1
    dissolve_count = grid_source.dissolve(by="index_right", aggfunc="count")
    grid_gdf.loc[dissolve_count.index, count_col] = dissolve_count[count_col].values

    # Calculate cumulative FRP if available
    if frp_available:
        frp_col = f'{source_name}_frp'
        grid_source[frp_col] = grid_source['frp'].fillna(0)
        dissolve_frp = grid_source.dissolve(by="index_right", aggfunc={frp_col: "sum"})
        grid_gdf.loc[dissolve_frp.index, frp_col] = dissolve_frp[frp_col].values

    return grid_gdf


if __name__ == "__main__":

    # Define the fire IDs of interest
    # 341, 345, 348 had other fires too close
    #fire_ids = ['2023_203', '2023_207', '2023_854', '2023_834', '2023_366', '2023_897', '2023_858']
    # 366 has better examples of hs inside/outside buffer
    fire_ids = ['2023_203']

    # Load the shapefile containing fire data
    nbac_shapefile = r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp'
    nbac = gpd.read_file(nbac_shapefile)

    # Convert the CRS to EPSG:4326 for compatibility with FIRMS data
    nbac = nbac.to_crs('EPSG:4326')

    # Filter for the fires of interest
    nbac_filtered = nbac[nbac['GID'].isin(fire_ids)].copy()
    nbac_filtered = nbac_filtered.reset_index()

    # Add a column with bounding box coordinates for each fire (based on the outside of the nbac)
    #nbac_filtered['bbox_coords'] = nbac_filtered['geometry'].apply(lambda geom: geom.bounds)

    # Determine utm zone for each fire (for buffering and projection)
    centroids = nbac_filtered['geometry'].centroid

    # Get lat and long
    lats = centroids.y
    lons = centroids.x

    # Define the UTM zone based on the centroids
    utm_zones = [utm.from_latlon(lat, lon)[2] for lat, lon in zip(lats, lons)]

    # Create a new CRS for each UTM zone
    crs_list = [CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': False}) for zone in utm_zones]

    # FIRMS API setup (replace with your own MAP_KEY)
    MAP_KEY = 'e865c77bb60984ab516517cd4cdadea0'
    url = f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}'

    # now let's check how many transactions we have
    # I exceeded my limit once so added this so I know why it fails sometimes
    import pandas as pd
    url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
    try:
      df = pd.read_json(url,  typ='series')
      print(df)
    except:
      # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
      print ("There is an issue with the query. \nTry in your browser: %s" % url)

    # Set buffer distances for each source
    viirs_buf_dist = 375*2.5
    landsat_buf_dist = 30*2.5
    modis_buf_dist = 1000*2.5
    # TODO: Understand how to get the GOES pixel size and write code that will automatically determine a
    # conversative buffer distance for the specified region
    goes_buf_dist = 2000*2.5

    # Loop over each fire in the filtered dataset
    for idx, fire in nbac_filtered.iterrows():

        # Get the appropriate UTM CRS for this fire based on its centroid
        crs = crs_list[idx]

        # Project the entire fire geometry to the corresponding UTM CRS (determined above)
        fire_gdf = gpd.GeoDataFrame([fire], geometry=[fire['geometry']],
                                    crs=nbac_filtered.crs)  # Create a temporary GeoDataFrame in 4326

        #fire_gdf = fire_gdf.drop('bbox_coords', axis=1)

        # project the temp gdf to the corresponding epsg for buffering
        fire_proj_viirs = fire_gdf.to_crs(crs.to_epsg())
        fire_proj_modis = fire_gdf.to_crs(crs.to_epsg())
        fire_proj_landsat = fire_gdf.to_crs(crs.to_epsg())
        fire_proj_goes = fire_gdf.to_crs(crs.to_epsg())

        # Create a buffer around the fire geometry
        fire_proj_viirs['geometry'] = fire_proj_viirs['geometry'].buffer(viirs_buf_dist)
        fire_proj_modis['geometry'] = fire_proj_modis['geometry'].buffer(modis_buf_dist)
        fire_proj_landsat['geometry'] = fire_proj_landsat['geometry'].buffer(landsat_buf_dist)
        fire_proj_goes['geometry'] = fire_proj_goes['geometry'].buffer(goes_buf_dist)

        # Convert back to the original CRS (WGS 84 / EPSG:4326)
        fire_buff_viirs = fire_proj_viirs.to_crs(epsg=4326)
        fire_buff_modis = fire_proj_modis.to_crs(epsg=4326)
        fire_buff_landsat = fire_proj_landsat.to_crs(epsg=4326)
        fire_buff_goes = fire_proj_goes.to_crs(epsg=4326)

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
        # TODO: Update so it uses the biggest buffer box (approx GOES atm)
        # It will probably be GOES eventually when I figure out that buffer

        # Add a column with bounding box coordinates for each fire (based on the outside of the nbac)
        bbox_coords = fire_buff_goes['geometry'].bounds
        bbox_coords = bbox_coords.reset_index(drop=True)
        coords_str = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"

        # Initialize dataframes for storing hotspots data
        df_SNPP_all = pd.DataFrame()
        df_MODIS_all = pd.DataFrame()
        df_GOES_all = pd.DataFrame()
        df_LS_all = pd.DataFrame()

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
        gdf_LS = gpd.GeoDataFrame(df_LS_all, geometry=gpd.points_from_xy(df_LS_all.longitude, df_LS_all.latitude), crs="EPSG:4326")

        all_hs = pd.concat([gdf_SNPP, gdf_MODIS, gdf_GOES, gdf_LS])

        # TODO: Determine which hotspots are inside the buffered perimeter and which are outside

        ## GRID DATA
        # https://james-brennan.github.io/posts/fast_gridding_geopandas/

        # Update this so it uses the goes bounds for now (with no buffer)
        xmin, ymin, xmax, ymax = fire_proj_goes.total_bounds

        # Define the number of grid cells
        n_cells = 30
        cell_size = (xmax - xmin) / n_cells  # Size of each grid cell

        # Create grid cells as shapely boxes
        grid_cells = []
        for x0 in np.arange(xmin, xmax, cell_size):
            for y0 in np.arange(ymin, ymax, cell_size):
                # Define grid cell bounds (x0, y0 is bottom-left corner)
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                grid_cells.append(box(x0, y0, x1, y1))

        # Convert grid cells to a GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=fire_proj_goes.crs)

        grid_gdf = grid_gdf.to_crs(epsg=4326)

        merged_all = gpd.sjoin(all_hs, grid_gdf, how='left', predicate='within')

        # make a simple count variable that we can sum
        merged_all['n_fires'] = 1
        # Compute stats per grid cell -- aggregate fires to grid cells with dissolve
        # dissolves observations into a single observation
        dissolve = merged_all.dissolve(by="index_right", aggfunc="count")
        # put this into cell
        grid_gdf.loc[dissolve.index, 'n_fires_all'] = dissolve.n_fires.values

        grid_gdf = process_gridding_and_frp(gdf_GOES, grid_gdf, 'goes')
        grid_gdf = process_gridding_and_frp(gdf_SNPP, grid_gdf, 'viirs')
        grid_gdf = process_gridding_and_frp(gdf_MODIS, grid_gdf, 'modis')
        grid_gdf = process_gridding_and_frp(gdf_LS, grid_gdf, 'ls', frp_available=False)

        grid_gdf = grid_gdf.fillna(0)
        df = pd.DataFrame(grid_gdf)
        df.to_csv(r'C:\Users\kzammit\Documents\DL-chapter\data.csv')


        print('test')

        #fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        #max_plot = grid_gdf['n_fires'].max()
        #grid_gdf.plot(column='n_fires', figsize=(12, 8), cmap='inferno',
        #              vmax=int(max_plot+1), edgecolor="grey", legend=True)
        #plt.autoscale(False)
        #ax.axis('off')
        #output_path = r'C:\Users\kzammit\Documents\DL-chapter\hotspots-fire-' + str(fire['GID']) + '-grid.png'
        #plt.savefig(output_path, bbox_inches='tight')

        #plot_fires()


