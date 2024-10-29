# This script plots all fires reported through NBAC and the NFDB for Sep 22/23 2023.
#
#

# =================================================================================
# Imports
# =================================================================================

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

# hs_sdate: date of the first detected hotspot within the spatial extent of the fire event
# hs_edate: date of the last detected hotspot within the spatial extent of the fire event
# ag_sdate: fire start date reported by the agency
# ag_edate: end date reported by the agency
# capdate: acquisition date of the source data
# poly_ha: total area calculated in hectares (canada albers equal area conic projection)
# adj_ha: adjusted area burn calculated in hectares
# gid: fire year and NFIREID concat

NBAC = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp')

date_format = '%Y/%m/%d'

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

# There are some cases where there is no agency date OR hotspot date
# Drop these
NBAC = NBAC.drop(NBAC[(NBAC.start_date == '0000/00/00')].index)
NBAC = NBAC.drop(NBAC[(NBAC.end_date == '0000/00/00')].index)

# Filter the hotspots so that we're only looking at fires which contain Sept 23 within their date range
date_obj = datetime.strptime('2023/09/23', date_format)

#NBAC['start_date'] = NBAC['start_date'].apply(pd.to_datetime)
NBAC['sept-23'] = 0
NBAC.loc[(datetime.strptime(NBAC['start_date'], date_format)) <= date_obj <= (datetime.strptime(NBAC['end_date'], date_format)), 'sept-23'] = 1

print('test')


