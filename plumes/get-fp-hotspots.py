import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from networkx import Graph, connected_components
from scipy.spatial import cKDTree


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

# Buffer distance
#buffer_dist = 375*math.sqrt(2)
buffer_dist = 375*2

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
        plt.savefig(r'C:\Users\kzammit\Documents\plumes\plots' + '\\' + type_str + '-' + str(fire) + '.png')
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

    ### Persistent heat sources
    pers_hs = gpd.read_file(pers_hs_shp)
    cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
    pers_hs_cad = pers_hs[pers_hs['prov'].isin(cad_provs)]

    ### Apply buffers to all sounds to account for hotspot resolution
    # NAD83 (EPSG 3978) is commonly used for Canada
    # Commenting this out because it takes a few minutes to run
    # print('Buffering boundaries')
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

    nbac_buff['perim-source'] = 'NBAC'
    nfdb_buff['perim-source'] = 'NFDB'
    pers_hs_cad_buff['perim-source'] = 'PH'

    df_perims = pd.concat([nbac_buff, nfdb_buff, pers_hs_cad_buff])

    nbac_doi = nbac_doi.to_crs(epsg=4326)
    nfdb_doi = nfdb_doi.to_crs(epsg=4326)

    ### Canada
    ne = gpd.read_file(nat_earth_shp)
    cad = ne[ne['ADMIN']=='Canada']

    # Create a bounding box around Canada (this will include some of the States, but we'll fix this later)
    bbox_coords = cad.bounds
    bbox_coords = bbox_coords.reset_index()
    coords = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"

    # Pull hotspots from FIRMS
    df_hotspots = fetch_viirs_hotspots(coords, doi_firms, cad)

    ### flag hotspots outside of boundaries
    tp_fp_flags = gpd.sjoin(df_hotspots, df_perims, predicate='within', how='left')

    col_index = tp_fp_flags.columns.get_loc('index_right0')
    tp_fp_flags_sub = tp_fp_flags.iloc[:, 0:col_index+1]

    tp_fp_flags_sub['NBAC-ID'] = tp_fp_flags['NFIREID']
    tp_fp_flags_sub['PH-ID'] = tp_fp_flags['gid']
    tp_fp_flags_sub['NFDB-ID'] = tp_fp_flags['NFDBFIREID']
    tp_fp_flags_sub['perim-source'] = tp_fp_flags['perim-source']

    tp_fp_flags_sub['Class'] = 1
    tp_fp_flags_sub.loc[tp_fp_flags_sub.index_right0.isnull(), 'Class'] = 0

    print('The number of false positives is ' + str(tp_fp_flags_sub.Class[tp_fp_flags_sub.Class == 0].count()))

    # Add a column that identifies the closest TP
    tp = tp_fp_flags_sub[tp_fp_flags_sub['Class']==1]
    fp = tp_fp_flags_sub[tp_fp_flags_sub['Class']==0]

    fp = fp.to_crs(epsg=3978)
    tp = tp.to_crs(epsg=3978)

    max_distance = 2000
    print('Determining closest perimeter within max distance of ' + str(max_distance))
    fp_w_flag = ckdnearest(fp, tp)
    fp_w_flag = fp_w_flag.dropna(axis=1, how='all')
    # reset the perim_source to none if farther than 2000 m
    fp_w_flag.loc[fp_w_flag['dist'] > max_distance, 'perim-source'] = "NONE"

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

    if len(none_fp) >= 1:

        print('Clustering fp > ' +str(max_distance) + ' from known perimeters')

        # currently this results in a lot of duplicates because some of the points are outside of the max distance
        # from eachother within the same cluster... how to get around this?

        # clump them together in minimum groups of 6 determined by the max distance
        none_fp = none_fp.to_crs(epsg=3978)
        none_fp = none_fp.reset_index()
        buffers = none_fp.geometry.buffer(max_distance)

        # Create an empty list to store intersections
        intersections = []

        # Perform pairwise intersection checks
        for i in range(len(buffers)):
            for j in range(i + 1, len(buffers)):  # Only check upper triangle (avoid duplicates)
                if buffers.iloc[i].intersects(buffers.iloc[j]):
                    intersections.append((i, j))

        # create a dataframe of intersections
        int_df = pd.DataFrame(intersections)

        # add another step here that checks if ANY of the points within the cluster are within the max distance
        # if the index val within a cluster is also within another cluster, merge them
        # Using networkx package in python which does network analysis
        graph = Graph()
        for _, group in int_df.groupby(0):
            values = group[1].tolist()
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    graph.add_edge(values[i], values[j])

        components = connected_components(graph)
        merged_groups = [set(component) for component in components]

        for idx, fire in enumerate(merged_groups):

            # get rows with these indices from none_fp
            df_temp = none_fp.loc[list(merged_groups[idx])]

            if len(df_temp) > 3:

                df_temp = df_temp.to_crs(epsg=3978)

                fig, ax = plt.subplots(figsize=(10, 8))
                df_temp.plot(ax=ax, color='red')
                plt.title('Cluster #: ' + str(idx))
                plt.savefig(r'C:\Users\kzammit\Documents\plumes\plots' + '\\' + 'cluster-' + str(idx) + '.png')
                plt.close()


    # TODO: Save clustered points with corresponding ID's into one dataframe
    # TODO: Add bounding box around the clusters for plotting imagery
    # TODO: Add a flag for each hot spot if it had a high scan angle
    # TODO: Add flag for if it's on the E or W of the closest TP



print('Done')

