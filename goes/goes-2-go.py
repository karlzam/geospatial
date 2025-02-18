from goes2go import GOES
from goes2go.data import goes_nearesttime
from datetime import datetime, timedelta
#import pandas as pd
#import boto3
#import botocore
from goes2go.data import goes_latest
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


g = goes_nearesttime(datetime(2020, 12, 25, 10),
                     satellite='goes16',
                     product='ABI',
                     return_as='xarray',
                     save_dir=r'C:\Users\kzammit\Documents\GOES')

# Download a GOES ABI dataset
#G = goes_latest(product='ABI')

# Make figure on Cartopy axes
#ax = plt.subplot(projection=G.rgb.crs)
ax = plt.subplot(projection=g.rgb.crs)
#ax.imshow(G.rgb.TrueColor(), **G.rgb.imshow_kwargs)
ax.imshow(g.rgb.TrueColor(), **g.rgb.imshow_kwargs)
ax.coastlines()
plt.savefig(r'C:\Users\kzammit\Documents\GOES\test.png')
