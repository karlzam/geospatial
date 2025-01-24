import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from networkx import Graph, connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint


###### User Inputs ######

year = '2023'

# Obtained from: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/
nbac_shp = r'C:\Users\kzammit\Documents\shp\nbac\nbac_' + str(year) + '_20240530.shp'

# At the time of writing this script, NFDB polygons were not available for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt
nfdb_shp = r'C:\Users\kzammit\Documents\shp\nfdb\NFDB_point_20240613.shp'

# Obtained from Piyush
pers_hs_shp = r'C:\Users\kzammit\Documents\shp\pers-hs\m3mask5_lcc.shp'

# Obtained from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/
# ne_10m_admin_0_countries.zip
nat_earth_shp = r'C:\Users\kzammit\Documents\shp\nat-earth\ne_10m_admin_0_countries.shp'

# Date to focus analysis on (training used 2023, val used 2022)
doi = str(year) + '/09/23'
yoi = int(year)

# FIRMS may key to use API
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'
doi_firms = str(year) + '-09-23'
doi_firms_2 = str(year) + '-09-24'

# Buffer distance
#buffer_dist = 375*math.sqrt(2)
buffer_dist = 375*3

shp_dir = r'C:\Users\kzammit\Documents\plumes\shp'
plot_output_dir = r'C:\Users\kzammit\Documents\plumes\plots'
df_dir = r'C:\Users\kzammit\Documents\plumes\dfs'

###### Code ######

# Helper function to project and buffer geometries
def project_and_buffer(gdf, target_epsg, buffer_dist=buffer_dist):
    gdf_projected = gdf.to_crs(target_epsg)
    gdf_projected['geometry'] = gdf_projected['geometry'].buffer(buffer_dist)
    gdf_projected = gdf_projected.to_crs(epsg=4326)
    return gdf_projected


def fetch_viirs_hotspots(coords, doi_firms, cad):
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


# From here: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    # Documentation explains this is a kd-tree for quick nearest neighbour lookup
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)

    # Keep the geometry column from gdA which is TP's
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
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


def plot_fp(fp_df, col_id, orig_id, type_str, buffer_df):
    for idx, fire in enumerate(fp_df[col_id].unique()):

        temp_df = fp_df[fp_df[col_id]== fire]
        perim = buffer_df[buffer_df[orig_id]==fire]

        temp_df = temp_df.to_crs(epsg=3978)
        perim = perim.to_crs(epsg=3978)

        fig, ax = plt.subplots(figsize=(10,8))
        perim.plot(ax=ax, color='black')
        temp_df.plot(ax=ax, color='red')
        plt.title(type_str + ' ID: ' + str(fire))
        plt.savefig(plot_output_dir + '\\' + type_str + '-' + str(fire) + '.png')
        plt.close()


if __name__ == "__main__":

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


    ### PERSISTENT HEAT SOURCES
    pers_hs = gpd.read_file(pers_hs_shp)
    cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
    pers_hs_cad = pers_hs[pers_hs['prov'].isin(cad_provs)]

    ### Apply buffers to all sounds to account for hotspot resolution
    # NAD83 (EPSG 3978) is commonly used for Canada
    # Commenting this out because it takes a few minutes to run
    #print('Buffering boundaries')
    #target_epsg = 3978
    #nbac_buff = project_and_buffer(nbac_doi, target_epsg)
    #nfdb_buff = project_and_buffer(nfdb_doi, target_epsg)
    #pers_hs_cad_buff = project_and_buffer(pers_hs_cad, target_epsg)
    #nbac_buff.to_file(shp_dir + '\\' + 'nbac-buff.shp')
    #nfdb_buff.to_file(shp_dir + '\\' + 'nfdb-buff.shp')
    #pers_hs_cad_buff.to_file(shp_dir + '\\' + 'pers-hs-cad-buff.shp')

    nbac_buff = gpd.read_file(shp_dir + '\\' + 'nbac-buff.shp')
    nfdb_buff = gpd.read_file(shp_dir + '\\' + 'nfdb-buff.shp')
    pers_hs_cad_buff = gpd.read_file(shp_dir + '\\' + 'pers-hs-cad-buff.shp')

    # Add source name to col called "perim-source" for ease of use later
    nbac_buff['perim-source'] = 'NBAC'
    nfdb_buff['perim-source'] = 'NFDB'
    pers_hs_cad_buff['perim-source'] = 'PH'

    # Append all sources together
    df_perims = pd.concat([nbac_buff, nfdb_buff, pers_hs_cad_buff])

    # Create a bounding box around Canada (this will include some of the States, but we'll fix this later)
    ne = gpd.read_file(nat_earth_shp)
    cad = ne[ne['ADMIN']=='Canada']
    bbox_coords = cad.bounds
    bbox_coords = bbox_coords.reset_index()
    coords = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"


    ### HOTSPOTS

    # Pull hotspots from FIRMS
    df_hotspots = fetch_viirs_hotspots(coords, doi_firms, cad)

    # Flag hotspots outside of boundaries and clean df
    tp_fp_flags = gpd.sjoin(df_hotspots, df_perims, predicate='within', how='left')
    col_index = tp_fp_flags.columns.get_loc('index_right0')
    tp_fp_flags_sub = tp_fp_flags.iloc[:, 0:col_index+1]
    tp_fp_flags_sub['NBAC-ID'] = tp_fp_flags['NFIREID']
    tp_fp_flags_sub['PH-ID'] = tp_fp_flags['gid']
    tp_fp_flags_sub['NFDB-ID'] = tp_fp_flags['NFDBFIREID']
    tp_fp_flags_sub['perim-source'] = tp_fp_flags['perim-source']

    # Set the class to 1 if it's a true positive, and 0 if it's a false positive (outside of the boundary)
    tp_fp_flags_sub['Class'] = 1
    tp_fp_flags_sub.loc[tp_fp_flags_sub.index_right0.isnull(), 'Class'] = 0

    # Add a column that identifies the closest TP
    tp = tp_fp_flags_sub[tp_fp_flags_sub['Class']==1]
    fp = tp_fp_flags_sub[tp_fp_flags_sub['Class']==0]
    print('The number of false positives is ' + str(len(fp)))

    # Determine the closest perimeter to each false positive (if less than max distance)
    # if > max distance, cluster points together according to max distance
    fp = fp.to_crs(epsg=3978)
    tp = tp.to_crs(epsg=3978)

    max_distance = 2000
    print('Determining closest perimeter within max distance of ' + str(max_distance))
    fp_w_flag = ckdnearest(fp, tp)
    fp_w_flag = fp_w_flag.dropna(axis=1, how='all')
    # reset the perim_source to none if farther than 2000 m
    fp_w_flag.loc[fp_w_flag['dist'] > max_distance, 'perim-source'] = "NONE"

    # Create sub dataframes for each source-type
    NBAC_fp = fp_w_flag[fp_w_flag['perim-source']=='NBAC']
    NBAC_fp = NBAC_fp.dropna(axis=1, how='all')

    NFDB_fp = fp_w_flag[fp_w_flag['perim-source']=='NFDB']
    NFDB_fp = NFDB_fp.dropna(axis=1, how='all')

    pers_hs_fp = fp_w_flag[fp_w_flag['perim-source']=='PH']
    pers_hs_fp = pers_hs_fp.dropna(axis=1, how='all')

    none_fp = fp_w_flag[fp_w_flag['perim-source']=='NONE']
    none_fp = none_fp.dropna(axis=1, how='all')

    print('Plotting NBAC, PH, NFDB, and clusters away from known perimeters')
    if len(NBAC_fp) >= 1:
        plot_fp(NBAC_fp, 'NBAC-ID', 'NFIREID', 'NBAC', nbac_buff)

    if len(pers_hs_fp) >= 1:
        plot_fp(pers_hs_fp, 'PH-ID', 'gid', 'pers-hs', pers_hs_cad_buff)

    if len(NFDB_fp) >= 1:
        plot_fp(NFDB_fp, 'NFDB-ID', 'NFDBFIREID', 'NFDB', nfdb_buff)


    # DBSCAN Clustering for the points with no boundary within the maximum distance
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

    # something is wrong here, the coordinates are swapped

    clusters_df['multipoint'] = clusters_df['points'].apply(array_to_multipoint)

    # Remove empty clusters that were indicated by DBSCAN
    clusters_df = clusters_df[clusters_df['multipoint'].apply(lambda x: not x.is_empty)]

    """
    # My old clustering algorithm
    
    cluster_df = gpd.GeoDataFrame(columns=none_fp.columns)

    cluster_df['cluster_id'] = 0
    if len(none_fp) >= 1:

        print('Clustering fp > ' +str(max_distance) + ' from known perimeters')

        # Buffer false positives in the "none" group by max distance
        none_fp = none_fp.to_crs(epsg=3978)
        none_fp = none_fp.reset_index()
        buffers = none_fp.geometry.buffer(max_distance)

        # Create an empty list to store intersections
        intersections = []

        # Perform pairwise intersection checks to see what fp's are close to each other
        for i in range(len(buffers)):
            for j in range(i + 1, len(buffers)):  # Only check upper triangle (avoid duplicates)
                if buffers.iloc[i].intersects(buffers.iloc[j]):
                    intersections.append((i, j))

        # create a dataframe of intersections
        int_df = pd.DataFrame(intersections)

        # if the index val within a cluster is also within another cluster, merge them
        # (this avoids duplicate clusters where points are > max distance individually, but where they actually
        # belong to the same cluster
        # Using networkx package in python which does network analysis
        graph = Graph()
        for _, group in int_df.groupby(0):
            values = group[1].tolist()
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    graph.add_edge(values[i], values[j])

        components = connected_components(graph)
        merged_groups = [set(component) for component in components]

        cols = none_fp.columns
        for idx, fire in enumerate(merged_groups):

            # get rows with these indices from none_fp
            df_temp = none_fp.loc[list(merged_groups[idx])]
            df_temp['cluster_id'] = idx
            cluster_df = pd.concat([cluster_df, df_temp])

            if len(df_temp) > 3:

                df_temp = df_temp.to_crs(epsg=3978)

                fig, ax = plt.subplots(figsize=(10, 8))
                df_temp.plot(ax=ax, color='red')
                plt.title('Cluster #: ' + str(idx))
                plt.savefig(plot_output_dir + '\\' + 'cluster-' + str(idx) + '.png')
                plt.close()
                
    """

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
        cluster_agg = cluster_agg.rename(columns={"index" : "id"})
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

    fp_all = pd.concat([NBAC_agg, cluster_agg, NFDB_agg, pers_hs_agg])
    fp_all = fp_all.reset_index(drop=True)

    # Create a new column 'bounds' and assign the bounds from the geometry column
    # Convert to 4326 for use in geemap
    fp_all['bounds'] = fp_all['geometry'].apply(lambda geom: geom.bounds)
    fp_all['start_date'] = doi_firms
    fp_all['end_date'] = doi_firms_2

    fp_all.to_excel(df_dir + '\\' + 'all-false-positives-dbscan.xlsx', index=False)

    # TODO: Add a flag for each hot spot if it had a high scan angle
    # TODO: Add flag for if it's on the E or W of the closest TP

print('Done')


