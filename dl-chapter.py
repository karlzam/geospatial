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
import math

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
    ca_map_proj.plot(ax=ax5, edgecolor='white', linewidth=1, color='black', zorder=1)
    fire_buff_goes.plot(ax=ax5, zorder=2, color='white')
    gdf_GOES.plot(ax=ax5, color='yellow', zorder=3)
    ax5.set_xlim(float(coords_str.split(',')[0]), float(coords_str.split(',')[2]))
    ax5.set_ylim(float(coords_str.split(',')[1]), float(coords_str.split(',')[3]))
    ax5.set_title('GOES HS, GOES Buffer')

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
    # This keeps the geometry column for the source_gdf, indicating what grid cell each hotspot belongs to with
    # the "within" keyword
    grid_source = gpd.sjoin(source_gdf, grid_gdf, how='left', predicate='within')

    # Add a column for hotspot counts
    count_col = f'{source_name}_hs'
    # Set the flag of "1" to indicate that row contains the specific type of hs
    grid_source[count_col] = 1
    # This counts the number of hot spots within each grid cell and sets all cols to that value
    dissolve_count = grid_source.dissolve(by="index_right", aggfunc="count")
    # Set the number of hotspots in each grid cell using the index and value from the dissolve_count dataframe
    grid_gdf.loc[dissolve_count.index, count_col] = dissolve_count[count_col].values

    # Calculate cumulative FRP if available
    # This uses the same method as above but sums the frp_col instead of doing a count of hs
    if frp_available:
        frp_col = f'{source_name}_frp'
        grid_source[frp_col] = grid_source['frp'].fillna(0)
        dissolve_frp = grid_source.dissolve(by="index_right", aggfunc={frp_col: "sum"})
        grid_gdf.loc[dissolve_frp.index, frp_col] = dissolve_frp[frp_col].values

    # oib = in/out buffer (true means outside buffer)
    oib_col = f'{source_name}_oib'
    grid_source[oib_col] = grid_source['out_buff']
    dissolve_oib = grid_source.dissolve(by="index_right", aggfunc={oib_col: "sum"})
    grid_gdf.loc[dissolve_oib.index, oib_col] = dissolve_oib[oib_col].values

    return grid_gdf


if __name__ == "__main__":

    # Define the fire IDs of interest
    # 341, 345, 348 had other fires too close
    #fire_ids = ['2023_203', '2023_207', '2023_854', '2023_834', '2023_366', '2023_897', '2023_858']
    # 366 has better examples of hs inside/outside buffer
    fire_ids = ['2023_203', '2023_854']

    # Load the shapefile containing fire data
    # TODO: Check if NBAC times are in UTC
    nbac_shapefile = r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp'
    nbac = gpd.read_file(nbac_shapefile)

    # Convert the CRS to EPSG:4326 for compatibility with FIRMS data
    nbac = nbac.to_crs('EPSG:4326')

    # Filter for the fires of interest
    nbac_filtered = nbac[nbac['GID'].isin(fire_ids)].copy()
    nbac_filtered = nbac_filtered.reset_index()

    # Use outside of NBAC perimeter (ie. created polygon boundary instead of multipolygon)
    # This breaks the bounding box statement, let's figure out why

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
    viirs_buf_dist = math.sqrt(2)*375
    landsat_buf_dist = math.sqrt(2)*30
    modis_buf_dist = math.sqrt(2)*1000
    # TODO: Something up with GOES buffer
    goes_buf_dist = math.sqrt(2)*4000

    # Loop over each fire in the filtered dataset
    all_data = gpd.GeoDataFrame()

    for idx, fire in nbac_filtered.iterrows():

        print('Working on fire id ' + str(fire['NFIREID']))

        # Get the appropriate UTM CRS for this fire based on its centroid
        crs = crs_list[idx]

        # TODO: This doesn't work for multipolygon but can't do it before or else it does all fires within the pull
        fire['geometry'] = fire['geometry'].union_all().convex_hull

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

        # Get the bounding box coordinates for the current fire using the largest perimeter
        # Add a column with bounding box coordinates for each fire (based on the outside of the nbac)
        # THIS DOES NOT WORK WITH POLYGON!!!! THIS WAS THE ERROR ALL ALONG :((((
        bbox_coords = fire_buff_modis['geometry'].bounds
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

        # This plots the 6 panel
        plot_fires()

        print('test')

        # Let's plot the hotspots on a map but colour by scan and track values
        # gdf_GOES.plot.scatter(x='scan', y='track')
        # gdf_GOES.plot(x='longitude', y='latitude', kind="scatter", color=gdf_GOES["track"])
        # gdf_GOES.plot(x='longitude', y='latitude', kind="scatter", color=gdf_GOES["scan"])

        print('test')

        # Add column for each dataframe that determines which spots are outside their corresponding buffers
        gdf_LS['out_buff'] = ~gdf_LS['geometry'].apply(lambda geom: fire_buff_landsat.contains(geom))
        print(gdf_LS.out_buff.value_counts())
        gdf_MODIS['out_buff'] = ~gdf_MODIS['geometry'].apply(lambda geom: fire_buff_modis.contains(geom))
        print(gdf_MODIS.out_buff.value_counts())
        gdf_SNPP['out_buff'] = ~gdf_SNPP['geometry'].apply(lambda geom: fire_buff_viirs.contains(geom))
        print(gdf_SNPP.out_buff.value_counts())
        gdf_GOES['out_buff'] = ~gdf_GOES['geometry'].apply(lambda geom: fire_buff_goes.contains(geom))
        print(gdf_GOES.out_buff.value_counts())

        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        #gdf_GOES.plot(ax=ax1, x='longitude', y='latitude', kind="scatter", color=gdf_GOES["out_buff"])
        #gdf_MODIS.plot(ax=ax2, x='longitude', y='latitude', kind="scatter", color=gdf_MODIS["out_buff"])
        #gdf_SNPP.plot(ax=ax3, x='longitude', y='latitude', kind="scatter", color=gdf_SNPP["out_buff"])
        #gdf_LS.plot(ax=ax4, x='longitude', y='latitude', kind="scatter", color=gdf_LS["out_buff"])
        #plt.savefig(r'C:\Users\kzammit\Documents\DL-chapter\203-outside-buff.png')

        ## GRID DATA
        # https://james-brennan.github.io/posts/fast_gridding_geopandas/

        # Update this so it uses the goes bounds for now (with no buffer)
        xmin, ymin, xmax, ymax = fire_proj_goes.total_bounds

        # Define the number of grid cells
        #n_cells = 30
        #cell_size = (xmax - xmin) / n_cells  # Size of each grid cell
        # Not sure if this is the right thing to do - look at other script for calculating pixel size based on math
        # and see if it's accurate
        cell_size = gdf_GOES['track'].max()

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

        # Currently forcing all frp avail to be false so it doesn't calculate that row
        # Piyush made a good point that right now I'm doing for the entire fire and for different sources so it
        # doesn't make a ton of sense to sum frp in this way
        grid_gdf = process_gridding_and_frp(gdf_GOES, grid_gdf, 'goes', frp_available=False)
        grid_gdf = process_gridding_and_frp(gdf_SNPP, grid_gdf, 'viirs', frp_available=False)
        grid_gdf = process_gridding_and_frp(gdf_MODIS, grid_gdf, 'modis', frp_available=False)
        grid_gdf = process_gridding_and_frp(gdf_LS, grid_gdf, 'ls', frp_available=False)
        grid_gdf['all_hs'] = grid_gdf[['goes_hs', 'viirs_hs', 'modis_hs', 'ls_hs']].sum(axis=1, min_count=1)

        #grid_gdf = grid_gdf.fillna(0)
        #df = pd.DataFrame(grid_gdf)
        #df.to_csv(r'C:\Users\kzammit\Documents\DL-chapter\data.csv')

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        max_plot = grid_gdf['all_hs'].max()
        grid_gdf.plot(column='all_hs', figsize=(12, 8), cmap='inferno',
                      vmax=int(max_plot+1), edgecolor="grey", legend=True)
        plt.autoscale(False)
        ax.axis('off')
        output_path = r'C:\Users\kzammit\Documents\DL-chapter\hotspots-fire-' + str(fire['GID']) + '-grid.png'
        plt.savefig(output_path, bbox_inches='tight')

        grid_gdf['goes_frac'] = grid_gdf['goes_oib']/grid_gdf['goes_hs']
        grid_gdf['viirs_frac'] = grid_gdf['viirs_oib'] / grid_gdf['viirs_hs']
        grid_gdf['modis_frac'] = grid_gdf['modis_oib'] / grid_gdf['modis_hs']
        grid_gdf['ls_frac'] = grid_gdf['ls_oib'] / grid_gdf['ls_hs']
        grid_gdf['all_frac'] = (grid_gdf[['goes_oib', 'viirs_oib', 'modis_oib', 'ls_oib']].sum(axis=1, min_count=1)/
                                grid_gdf['all_hs'])

        grid_gdf = grid_gdf.dropna(axis=0, how='all', subset=grid_gdf.columns[1:])
        grid_clean = grid_gdf.fillna(0)

        all_data = pd.concat([all_data, grid_clean])

    print('test')



