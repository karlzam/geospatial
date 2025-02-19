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

end_time = utc_time + timedelta(hours=16)
start = utc_time.strftime("%Y-%m-%d %H:%M")
end = end_time.strftime("%Y-%m-%d %H:%M")

#g = goes_timerange(start, end,
#                   satellite='goes16',
#                   product='ABI',
#                   return_as='filelist', save_dir=r'C:\Users\kzammit\Documents\GOES')

g = goes_timerange(start, end,
                   satellite='goes16',
                   product='ABI',
                   return_as='xarray', save_dir=r'C:\Users\kzammit\Documents\GOES')


print('test')

