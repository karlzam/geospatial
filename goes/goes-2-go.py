from goes2go import GOES
from goes2go.data import goes_nearesttime
from goes2go.data import goes_timerange
from datetime import datetime, timedelta
from goes2go.data import goes_latest
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from timezonefinder import TimezoneFinder
import pytz
import xarray as xr
import os
import glob
from PIL import Image
import re
import metpy
import numpy as np

# later draw the box on top of the file
fires = pd.read_excel(r'C:\Users\kzammit\Documents\Plumes\all-false-positives-manual-2023-09-23.xlsx')

# get the first coordinate from "bounds" (which is in EPSG4326 from the script)
# check what the conversion is for that timezone from UTC
# figure out what time 1:30 pm local is in UTC

# coord is long, lat
coord_1 = fires['bounds'][0].split(',')[0].split('(')[1]
coord_2 = fires['bounds'][0].split(',')[1].split(' ')[1]

# get UTC time for 1:00 pm local
tf = TimezoneFinder()
timezone = tf.timezone_at(lng=float(coord_1), lat=float(coord_2))
local_tz = pytz.timezone(timezone)
local_time = local_tz.localize(datetime(2023, 9, 23, 10, 0))  # 1 PM
utc_time = local_time.astimezone(pytz.UTC)

print(f"Local Time: {local_time}")
print(f"UTC Time: {utc_time}")

# set up the commands to use for goes2go
end_time = utc_time + timedelta(hours=16)
start = utc_time.strftime("%Y-%m-%d %H:%M")
end = end_time.strftime("%Y-%m-%d %H:%M")

# download the files for goes2go and save the info in a dataframe (file location, name, time, etc - very handy!)
g = goes_timerange(start, end,
                   satellite='goes16',
                   product='ABI',
                   return_as='filelist', save_dir=r'C:\Users\kzammit\Documents\GOES')

# for each 5th file, let's create some imagery to make gifs with
main_dir = r'C:\Users\kzammit\Documents\GOES'
img_dir = os.path.join(main_dir, 'images')
for idx, file in enumerate(g.iterrows()):
    if idx % 5 == 0:

        # open the nc file that was pulled with goes2go
        nc_file = xr.open_dataset(os.path.join(main_dir, file[1]['file']))
        ax = plt.subplot(projection=nc_file.rgb.crs)
        ax.imshow(nc_file.rgb.TrueColor(), **nc_file.rgb.imshow_kwargs)
        ax.coastlines()
        plt.savefig(os.path.join(img_dir, 'test' + str(idx) + '.png'))

        print('test')

# make an animated gif
gif_dir = os.path.join(main_dir, 'gifs')
image_files = glob.glob(os.path.join(img_dir, '*.png'))
files_sorted = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))
images = [Image.open(img) for img in files_sorted]
images[0].save(os.path.join(gif_dir, 'animated.gif'), save_all=True, append_images=images[1:], duration=200, loop=0)

