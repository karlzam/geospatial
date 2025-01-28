"""

get-fp-hotspots.py

This script investigates hotspot detections from the VIIRS instrument. It:
1. Loads NBAC, NFDB, and persistent hotspot polygons for a date of interest
2. Buffers polygons by a set distance
3. Removes inner islands from buffered polygons
4. Grabs VIIRS FIRMS hotspots for all of Canada
5. Labels hotspots as true or false positive if within a known buffered perimeter
6. Assigns fire ID (NBAC, NFDB, or persistent hotspot) to false positives within maximum distance from the closest
perimeter
7. Clusters fp's using DBScan (min points 3) not within maximum distance to a known perimeter
8. Exports csv file with relevant information for use in geemap script (inspect-fp.ipynb)

Author: Karlee Zammal the Party Mammal
Contact: karlee.zammit@nrcan-rncan.gc.ca
Date: 2025-01-25

"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, MultiPolygon, Polygon


###### User Inputs ######

# Date of Interest
# dois = ['2023/09/18', '2023/09/19', '2023/09/20','2023/09/21', '2023/09/22', '2023/09/23',
#        '2023/09/24', '2023/09/25', '2023/09/26', '2023/09/27', '2023/09/28', '2023/09/29']
dois = ['2023/09/22', '2023/09/23', '2023/09/24']

# Buffer Distance
# This distance is applied to the NBAC, NFDB, and persistent hotspot polygons
buffer_dist = 375 * 3

# Max Distance
# The max distance to consider points near boundaries or to cluster points
max_distance = 2000

# NBAC Folder
# Where the nbac shapefiles live
nbac_folder = r'C:\Users\kzammit\Documents\shp\nbac'

# NFDB Folder and Polygon
# At the time of writing this script, NFDB polygons were not available for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt
nfdb_folder = r'C:\Users\kzammit\Documents\shp\nfdb'
nfdb_shp = 'NFDB_point_20240613.shp'

# Persistent Hotspots
# File obtained from Piyush
pers_hs_shp = r'C:\Users\kzammit\Documents\shp\pers-hs\m3mask5_lcc.shp'

# Natural Earth Shapefile
# Obtained from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/
# ne_10m_admin_0_countries.zip
nat_earth_shp = r'C:\Users\kzammit\Documents\shp\nat-earth\ne_10m_admin_0_countries.shp'

# Directories
shp_dir = r'C:\Users\kzammit\Documents\plumes\shp'
plot_output_dir = r'C:\Users\kzammit\Documents\plumes\plots'
df_dir = r'C:\Users\kzammit\Documents\plumes\dfs'

# FIRMS Map Key
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'


###### Functions ######


def project_and_buffer(gdf, target_epsg, buffer_dist):
    """
    Project and buffer geometries to epsg 4326
    :param gdf: geodataframe to buffer/project
    :param target_epsg: the epsg to use for buffering
    :param buffer_dist: distance in m to buffer the polygons by
    :return:
    """
    gdf_projected = gdf.to_crs(target_epsg)
    gdf_projected['geometry'] = gdf_projected['geometry'].buffer(buffer_dist)
    gdf_projected = gdf_projected.to_crs(epsg=4326)
    return gdf_projected


def fetch_viirs_hotspots(coords, doi_firms, cad):
    """
    Grab viirs hotspots (both NOOA20 & NOAA21) from FIRMS and crop to include only within Canada
    :param FIRMS_map_key: Your personal FIRMS key for using their API
    :param coords: region to grab hotspots within
    :param doi_firms: date of interest to grab hotspots from (currently only pulls 1 day at a time by design)
    :param cad: canadian shape file to remove hotspots pulled outside of Canada due to rectangular box
    :return:
    """

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

    ### Concat hotspots
    gdf_SNPP['sat'] = 'SNPP'
    gdf_NOAA20['sat'] = 'NOAA20'
    df_hotspots = pd.concat([gdf_SNPP, gdf_NOAA20])

    return df_hotspots


def kdnearest(gdA, gdB):
    """
    Determine the closest possible perimeter by referencing the tp database and grabbing the source name
    :param gdA: geodataframe a (in this case, fp)
    :param gdB: geodataframe b (in this case, tp)
    :return:
    """
    # nA = fp
    # nB = tp
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))

    # https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.spatial.KDTree.html
    btree = KDTree(nB)
    dist, idx = btree.query(nA, k=1)

    # Keep the geometry column from gdA which is TP's
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)

    # I use these later
    cols_to_append = ['perim-source', 'NBAC-ID', 'NFDB-ID', 'PH-ID']

    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            pd.Series(dist, name='dist'),
            pd.Series(idx, name='idx'),
            gdB_nearest[cols_to_append],
        ],
        axis=1)

    return gdf


def plot_fp(fp_df, col_id, orig_id, type_str, buffer_df, date_label):
    """
    Function to plot perimeters with corresponding false positives indicated within the maximum distance
    :param fp_df: false positive dataframe
    :param col_id: what column to look in to grab the fire id
    :param orig_id: original fire id
    :param type_str: what source type the cluster belongs to (NBAC, NFDB, etc)
    :param buffer_df: perimeter buffer dataframe
    :param date_label: what date is being plotted
    :return:
    """

    for idx, fire in enumerate(fp_df[col_id].unique()):

        temp_df = fp_df[fp_df[col_id]== fire]
        perim = buffer_df[buffer_df[orig_id]==fire]

        temp_df = temp_df.to_crs(epsg=3978)
        perim = perim.to_crs(epsg=3978)

        fig, ax = plt.subplots(figsize=(10,8))
        perim.plot(ax=ax, color='black')
        temp_df.plot(ax=ax, color='red')
        plt.title(type_str + ' ID: ' + str(fire))
        plt.savefig(plot_output_dir + '\\' + type_str + '-' + str(fire) + '-' + str(date_label) + '.png')
        plt.close()


def remove_holes_from_geometries(geodataframe):
    """
    Remove holes from all polygons in the geometry column of a GeoDataFrame.

    Parameters:
        geodataframe (gpd.GeoDataFrame): Input GeoDataFrame with a geometry column.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with polygons having no holes.
    """

    def extract_exterior(geometry):
        if geometry is None:
            return None
        if geometry.geom_type == "Polygon":
            # Return a new Polygon using only the exterior coordinates
            return Polygon(geometry.exterior)
        elif geometry.geom_type == "MultiPolygon":
            # Create a new MultiPolygon from the exteriors of all polygons
            return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])
        else:
            # Return the geometry as-is if it's not a Polygon or MultiPolygon
            return geometry

    # Apply the function to the geometry column
    geodataframe['geometry'] = geodataframe['geometry'].apply(extract_exterior)
    return geodataframe


def concat_perims(agg_df, buff_df, flag):
    """
    Join together NBAC polygons that have the same NFIREID (applied to NBAC data)
    :param agg_df: final dataframe to write to (nbac_agg)
    :param buff_df: nbac_4326
    :param flag: column to pull the id from (NFIREID for NBAC)
    :return:
    """
    concattd_perims = []
    unique_ids = agg_df['id'].unique()
    for id_fi in unique_ids:
        temp = buff_df[buff_df[flag] == id_fi]
        if len(temp) == 1:
            temp = temp.reset_index()
            concattd_perims.append(temp['geometry'].loc[0])
        if len(temp) > 1:
            concattd_perims.append(temp['geometry'].union_all())

    return concattd_perims


if __name__ == "__main__":

    # get years from the date of interest
    years = [doi.split('/')[0] for doi in dois]

    # formatting dates for firms
    dois_firms = ['-'.join(doi.split('/')) for doi in dois]
    dois_firms_2 = ['-'.join(doi.split('/')[:2]) + '-' + f"{int(doi.split('/')[2]) + 1:02d}" for doi in dois]

    # set up arrays for nbac and nfdb (repeats for each day of interest due to how script works)
    nbac_shps = [nbac_folder + '\\' + 'nbac_' + doi.split('/')[0] + '_20240530.shp' for doi in dois]
    nfdb_shps = [nfdb_folder + '\\' + nfdb_shp for _ in dois]

    # For each date of interest (referencing each one by year even though it's just the first part of each doi)
    for idx, year in enumerate(years):

        # Set up date
        doi = dois[idx]
        yoi = int(year)
        doi_firms = dois_firms[idx]
        doi_firms_2 = dois_firms_2[idx]

        # Read NBAC file
        nbac = gpd.read_file(nbac_shps[idx])

        # NBAC Date Formatting
        # NBAC has agency start date as well as hotspot start date, and these often do not align
        # If one of them is missing, the fill value is "0000/00/00"
        # Create start and end date columns from the earlier or later of the two respectively (ignoring the fill val)

        nbac['start_date'] = 'tbd'

        # If either agency or hotspot start date is empty, assign the other as the start date
        nbac.loc[nbac['HS_SDATE'] == '0000/00/00', 'start_date'] = nbac['AG_SDATE']
        nbac.loc[nbac['AG_SDATE'] == '0000/00/00', 'start_date'] = nbac['HS_SDATE']

        # Pick the earlier of the two for the start date, excluding the empties
        nbac.loc[(nbac['HS_SDATE'] <= nbac['AG_SDATE']) & (nbac['HS_SDATE'] != '0000/00/00'), 'start_date'] = nbac[
            'HS_SDATE']
        nbac.loc[(nbac['AG_SDATE'] <= nbac['HS_SDATE']) & (nbac['AG_SDATE'] != '0000/00/00'), 'start_date'] = nbac[
            'AG_SDATE']

        # Do the same steps for the end date
        nbac['end_date'] = 'tbd'
        nbac.loc[nbac['HS_EDATE'] == '0000/00/00', 'end_date'] = nbac['AG_EDATE']
        nbac.loc[nbac['AG_EDATE'] == '0000/00/00', 'end_date'] = nbac['HS_EDATE']
        nbac.loc[(nbac['HS_EDATE'] >= nbac['AG_EDATE']) & (nbac['HS_EDATE'] != '0000/00/00'), 'end_date'] = nbac[
            'HS_EDATE']
        nbac.loc[(nbac['AG_EDATE'] >= nbac['HS_EDATE']) & (nbac['AG_EDATE'] != '0000/00/00'), 'end_date'] = nbac[
            'AG_EDATE']

        # There are some cases where there is no agency date OR hotspot date, drop these
        nbac = nbac.drop(nbac[(nbac.start_date == '0000/00/00')].index)
        nbac = nbac.drop(nbac[(nbac.end_date == '0000/00/00')].index)

        # Filter the hotspots so we're only looking at fires which contain Sept 23 within their date range
        date_format = '%Y/%m/%d'
        date_obj = datetime.strptime(doi, date_format)
        nbac['doi'] = 0
        nbac = nbac.assign(start_dt=lambda x: pd.to_datetime(x['start_date'], format=date_format))
        nbac = nbac.assign(end_dt=lambda x: pd.to_datetime(x['end_date'], format=date_format))
        nbac.loc[(nbac['start_dt'] <= date_obj) & (nbac['end_dt'] >= date_obj), "doi"] = 1

        # Create a new dataframe for only fires containing the date of interest
        nbac_doi = nbac[nbac['doi'] == 1]
        nbac_doi = nbac_doi.reset_index()

        ### NFDB (similar steps)
        nfdb = gpd.read_file(nfdb_shps[idx])
        nfdb_doi = nfdb[nfdb['YEAR'] == yoi]
        nfdb_doi = nfdb_doi.drop(nfdb_doi[(nfdb_doi.OUT_DATE == '0000/00/00')].index)
        nfdb_doi['doi'] = 0
        nfdb_doi = nfdb_doi.assign(start_dt=lambda x: pd.to_datetime(x['REP_DATE'], format=date_format))
        nfdb_doi = nfdb_doi.assign(end_dt=lambda x: pd.to_datetime(x['OUT_DATE'], format=date_format))
        nfdb_doi.loc[(nfdb_doi['start_dt'] <= date_obj) & (nfdb_doi['end_dt'] >= date_obj), "doi"] = 1
        nfdb_doi = nfdb_doi[nfdb_doi['doi'] == 1]
        nfdb_doi = nfdb_doi.reset_index()

        ### PERSISTENT HEAT SOURCES
        pers_hs = gpd.read_file(pers_hs_shp)
        cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
        pers_hs_cad = pers_hs[pers_hs['prov'].isin(cad_provs)]

        # Buffer boundaries
        print('Buffering boundaries')
        target_epsg = 3978
        nbac_buff = project_and_buffer(nbac_doi, target_epsg, buffer_dist)
        nfdb_buff = project_and_buffer(nfdb_doi, target_epsg, buffer_dist)
        pers_hs_cad_buff = project_and_buffer(pers_hs_cad, target_epsg, buffer_dist)

        # remove inner islands from buffered perimeters
        nbac_buff_filled = remove_holes_from_geometries(nbac_buff)

        # save files to shapefile dir
        nbac_buff_filled.to_file(shp_dir + '\\' + 'nbac-buff-filled' + str(doi_firms) + '.shp')
        nfdb_buff.to_file(shp_dir + '\\' + 'nfdb-buff-' + str(doi_firms) + '.shp')
        pers_hs_cad_buff.to_file(shp_dir + '\\' + 'pers-hs-cad-buff-' + str(doi_firms) + '.shp')

        # concat all perimeters into one dataframe for ease of use
        nbac_buff_filled['perim-source'] = 'NBAC'
        nfdb_buff['perim-source'] = 'NFDB'
        pers_hs_cad_buff['perim-source'] = 'PH'
        df_perims = pd.concat([nbac_buff_filled, nfdb_buff, pers_hs_cad_buff])

        # Create a bounding box around Canada (this will include some of the States, but we'll fix this later)
        ne = gpd.read_file(nat_earth_shp)
        cad = ne[ne['ADMIN'] == 'Canada']
        bbox_coords = cad.bounds
        bbox_coords = bbox_coords.reset_index()
        coords = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"

        print('Pulling hotspots')
        # Pull hotspots from FIRMS
        df_hotspots = fetch_viirs_hotspots(coords, doi_firms, cad)

        # Flag hotspots outside of boundaries and clean df
        tp_fp_flags = gpd.sjoin(df_hotspots, df_perims, predicate='within', how='left')
        col_index = tp_fp_flags.columns.get_loc('index_right0')

        # THIS LOOKS NORMAL
        tp_fp_flags_sub = tp_fp_flags.iloc[:, 0:col_index + 1]

        # Clean up the labelling for consistency
        tp_fp_flags_sub['NBAC-ID'] = tp_fp_flags['NFIREID']
        tp_fp_flags_sub['PH-ID'] = tp_fp_flags['gid']
        tp_fp_flags_sub['NFDB-ID'] = tp_fp_flags['NFDBFIREID']
        tp_fp_flags_sub['perim-source'] = tp_fp_flags['perim-source']

        # Set the class to 1 if it's a true positive, and 0 if it's a false positive (outside of the boundary)
        tp_fp_flags_sub['Class'] = 1
        tp_fp_flags_sub.loc[tp_fp_flags_sub.index_right0.isnull(), 'Class'] = 0

        # Add a column that identifies the closest TP
        tp = tp_fp_flags_sub[tp_fp_flags_sub['Class'] == 1]
        fp = tp_fp_flags_sub[tp_fp_flags_sub['Class'] == 0]
        print('The number of false positives is ' + str(len(fp)))

        # Determine the closest perimeter to each false positive (if less than max distance)
        # if > max distance, cluster points together according to max distance
        fp = fp.to_crs(epsg=3978)
        tp = tp.to_crs(epsg=3978)
        print('Determining closest perimeter within max distance of ' + str(max_distance))
        fp_w_flag = kdnearest(fp, tp)
        fp_w_flag = fp_w_flag.dropna(axis=1, how='all')
        # reset the perim_source to none if farther than 2000 m
        fp_w_flag.loc[fp_w_flag['dist'] > max_distance, 'perim-source'] = "NONE"

        # Create sub dataframes for each source-type
        NBAC_fp = fp_w_flag[fp_w_flag['perim-source'] == 'NBAC']
        NBAC_fp = NBAC_fp.dropna(axis=1, how='all')

        NFDB_fp = fp_w_flag[fp_w_flag['perim-source'] == 'NFDB']
        NFDB_fp = NFDB_fp.dropna(axis=1, how='all')

        pers_hs_fp = fp_w_flag[fp_w_flag['perim-source'] == 'PH']
        pers_hs_fp = pers_hs_fp.dropna(axis=1, how='all')

        none_fp = fp_w_flag[fp_w_flag['perim-source'] == 'NONE']
        none_fp = none_fp.dropna(axis=1, how='all')

        print('Plotting NBAC, PH, NFDB, and clusters away from known perimeters')
        if len(NBAC_fp) >= 1:
            #plot_fp(NBAC_fp, 'NBAC-ID', 'NFIREID', 'NBAC', nbac_buff_conc, doi_firms)
            plot_fp(NBAC_fp, 'NBAC-ID', 'NFIREID', 'NBAC', nbac_buff_filled, doi_firms)

        if len(pers_hs_fp) >= 1:
            plot_fp(pers_hs_fp, 'PH-ID', 'gid', 'pers-hs', pers_hs_cad_buff, doi_firms)

        if len(NFDB_fp) >= 1:
            plot_fp(NFDB_fp, 'NFDB-ID', 'NFDBFIREID', 'NFDB', nfdb_buff, doi_firms)

        print('DBScan Clustering')
        # DBScan Clustering
        # Note that the crs for the undefined clusters will be EPSG:4326 because the shapely geometry column is the only
        # col affected by "to_crs"
        coords = none_fp[['longitude', 'latitude']].to_numpy()
        kms_per_radian = 6371.0088
        epsilon = 2 / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
        clusters_df = {"points": clusters}
        clusters_df = pd.DataFrame(clusters_df)

        def array_to_multipoint(array):
            return MultiPoint(array)

        clusters_df['multipoint'] = clusters_df['points'].apply(array_to_multipoint)
        # Remove empty clusters that were indicated by DBSCAN
        clusters_df = clusters_df[clusters_df['multipoint'].apply(lambda x: not x.is_empty)]

        # Create one dataframe for all aggregated fp's
        NBAC_agg = gpd.GeoDataFrame(columns=NBAC_fp.columns)
        if len(NBAC_fp) > 0:
            NBAC_agg = NBAC_fp[['geometry', 'NBAC-ID']]
            NBAC_agg = NBAC_agg.dissolve(by='NBAC-ID', aggfunc='sum')
            NBAC_agg = NBAC_agg.reset_index()
            NBAC_agg['source'] = 'NBAC'
            NBAC_agg = NBAC_agg.rename(columns={"NBAC-ID": "id"})
            NBAC_agg = NBAC_agg.to_crs('EPSG:4326')

        cluster_agg = gpd.GeoDataFrame(columns=clusters_df.columns)
        if len(none_fp) > 0:
            cluster_agg = gpd.GeoDataFrame(clusters_df, geometry='multipoint', crs="EPSG:4326")
            cluster_agg = cluster_agg.reset_index()
            cluster_agg = cluster_agg.rename(columns={"index": "id"})
            cluster_agg = cluster_agg.dissolve(by='id', aggfunc='sum')
            cluster_agg = cluster_agg.reset_index()
            cluster_agg['source'] = 'None'
            cluster_agg = cluster_agg.drop(['points'], axis=1)
            cluster_agg = cluster_agg.to_crs('EPSG:4326')
            cluster_agg = cluster_agg.rename(columns={"multipoint": "geometry"})

        NFDB_agg = gpd.GeoDataFrame(columns=NFDB_fp.columns)
        if len(NFDB_fp) > 0:
            NFDB_agg = NFDB_fp[['geometry', 'NFDB-ID']]
            NFDB_agg = NFDB_agg.dissolve(by='NFDB-ID', aggfunc='sum')
            NFDB_agg = NFDB_agg.reset_index()
            NFDB_agg['source'] = 'NFDB'
            NFDB_agg = NFDB_agg.rename(columns={"NFDB-ID": "id"})
            NFDB_agg = NFDB_agg.to_crs('EPSG:4326')

        pers_hs_agg = gpd.GeoDataFrame(columns=pers_hs_fp.columns)
        if len(pers_hs_fp) > 0:
            pers_hs_agg = pers_hs_fp[['geometry', 'PH-ID']]
            pers_hs_agg = pers_hs_agg.dissolve(by='PH-ID', aggfunc='sum')
            pers_hs_agg = pers_hs_agg.reset_index()
            pers_hs_agg['source'] = 'PH'
            pers_hs_agg = pers_hs_agg.rename(columns={"PH-ID": "id"})
            pers_hs_agg = pers_hs_agg.to_crs('EPSG:4326')

        # concatenate perimeters together because NBAC sometimes uses the same fire ID for multiple rows
        if len(NBAC_agg) > 0:
            #nbac_4326 = nbac_buff_conc.to_crs("EPSG:4326")
            nbac_4326 = nbac_buff_filled.to_crs("EPSG:4326")
            NBAC_agg['fire-perimeter'] = concat_perims(NBAC_agg, nbac_4326, 'NFIREID')

        # Concat all dataframes together
        fp_all = pd.concat([NBAC_agg, cluster_agg, NFDB_agg, pers_hs_agg])
        fp_all = fp_all.reset_index(drop=True)

        # Create a new column 'bounds' and assign the bounds from the geometry column
        # Convert to 4326 for use in geemap
        fp_all['bounds'] = fp_all['geometry'].apply(lambda geom: geom.bounds)
        fp_all['start_date'] = doi_firms
        fp_all['end_date'] = doi_firms_2

        #fp_all.to_excel(df_dir + '\\' + 'all-false-positives-' + str(dois_firms[idx]) + '.xlsx', index=False)
        fp_all.to_csv(df_dir + '\\' + 'all-false-positives-' + str(dois_firms[idx]) + '.csv', index=False)

        print('Done ' + str(dois_firms[idx]))
