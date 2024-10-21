from goes2go.data import goes_nearesttime
from goes2go.data import goes_timerange

from datetime import datetime, timedelta
import pandas as pd

# GOES-2-GO has GOES-East/16 and GOES-West/17/18
# List of products: https://github.com/blaylockbk/goes2go/blob/main/goes2go/product_table.txt
# Info on satellite data processing levels: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_processing_levels.php#:~:text=Level%201b%20data%20are%20used,their%20Level%201b%20source%20data.
# Level 1B Data: Radiances measured by a satellite sensor that have been calibrated and geo-located
# (measured electromagnetic radiation at specific wavelengths)
# Level 2 Data: Level 1B data used in an algorithm to derive geophysical parameters, with the same spatial resolution
# and observation frequency as
# their Level1B source data
# Level 3 Data: Level 2 data that has been mapped to a uniform space-time grid, and therefore has been averaged over
# space and/or time.

# Info packet on GOES: https://noaa-goes16.s3.amazonaws.com/Beginners_Guide_to_GOES-R_Series_Data.pdf

# Goes-2-go pulls data from AWS: https://registry.opendata.aws/noaa-goes/
# https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16

# ABI: Advanced Baseline Imager
# CONUS: Continental US
# GLM: Geostationary Lightning Mapper

# Bands technical summary chart: https://www.goes-r.gov/spacesegment/ABI-tech-summary.html

#g = goes_nearesttime(datetime(2020, 12, 25, 10),
#                     satellite='goes16',
#                     product='ABI-L2-RSRC',
#                     return_as='xarray')

g = goes_nearesttime(datetime(2019, 10, 27, 10),
                     satellite='goes16',
                     product='ABI-L1b-RadC', download=True, overwrite=False,
                    save_dir=r'C:\Users\kzammit\Documents\GOES', verbose=True)


# This doesn't work! Something to do with frozen variables not working properly
# https://blaylockbk.github.io/goes2go/_build/html/reference_guide/index.html#goes2go.data.goes_timerange
# https://github.com/blaylockbk/goes2go/blob/main/goes2go/product_table.txt
# This only works if download is true
#g2 = goes_timerange(start='2019-10-27', end='2019-10-29', satellite = 'EAST', product='ABI-L1b-Rad', domain='C',
#                    return_as='xarray', download=True, max_cpus=1, overwrite=False,
#                    save_dir=r'C:\Users\kzammit\Documents\GOES', verbose=True)

print('test')



