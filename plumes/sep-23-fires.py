# This script plots all fires reported through NBAC and the NFDB for Sep 22/23 2023.
#
#

# =================================================================================
# Imports
# =================================================================================

import pandas as pd
import geopandas as gpd
import numpy as np

# hs_sdate: date of the first detected hotspot within the spatial extent of the fire event
# hs_edate: date of the last detected hotspot within the spatial extent of the fire event
# ag_sdate: fire start date reported by the agency
# ag_edate: end date reported by the agency
# capdate: acquisition date of the source data
# poly_ha: total area calculated in hectares (canada albers equal area conic projection)
# adj_ha: adjusted area burn calculated in hectares
# gid: fire year and NFIREID concat

NBAC = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp')


# The hotspot and agency start/end dates do not always align, so need to make if statement and grab the earlier
# of the two for the start date and the later of the two for the end date, while also accounting for the '0000/00/00'
# if there was no date reported (I'm assuming)
NBAC['start_date'] = 'tbd'
NBAC.loc[NBAC['HS_SDATE'] == '0000/00/00', 'start_date'] = NBAC['AG_SDATE']
NBAC.loc[NBAC['AG_SDATE'] == '0000/00/00', 'start_date'] = NBAC['HS_SDATE']
NBAC.loc[(NBAC['HS_SDATE'] <= NBAC['AG_SDATE']) & (NBAC['HS_SDATE'] != '0000/00/00'), 'start_date'] = NBAC['HS_SDATE']
NBAC.loc[(NBAC['AG_SDATE'] <= NBAC['HS_SDATE']) & (NBAC['AG_SDATE'] != '0000/00/00'), 'start_date'] = NBAC['AG_SDATE']

NBAC['end_date'] = 'tbd'
NBAC.loc[NBAC['HS_EDATE'] == '0000/00/00', 'end_date'] = NBAC['AG_EDATE']
NBAC.loc[NBAC['AG_EDATE'] == '0000/00/00', 'end_date'] = NBAC['HS_EDATE']
NBAC.loc[(NBAC['HS_EDATE'] >= NBAC['AG_EDATE']) & (NBAC['HS_EDATE'] != '0000/00/00'), 'end_date'] = NBAC['HS_EDATE']
NBAC.loc[(NBAC['AG_EDATE'] >= NBAC['HS_EDATE']) & (NBAC['AG_EDATE'] != '0000/00/00'), 'end_date'] = NBAC['AG_EDATE']

# Filter the hotspots so that we're only looking at fires which contain Sept 23 within their date range

print('test')


