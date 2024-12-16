#############################################################################
# CFSDB - Estimate day of burning and extract covariates (Python Version)
# Converted from R by [Your Name], December 2024
#############################################################################

# Required libraries
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Set working directory
import os
os.chdir(r"C:\Users\kzammit\Documents\CFSDS\CFSDS_example_Nov2023")

# 1.0 Processing --------------------------------------------------
# Load base grid (must be in projected coordinate system, e.g., EPSG:9001)
basegrid = rasterio.open("basegrid_180m.tif")

# 1.1 Process perimeter
perimeter = gpd.read_file("perimeter.shp")
perimeter["STARTDATE"] = pd.to_datetime(perimeter["AFSDATE"])
perimeter["ENDDATE"] = pd.to_datetime(perimeter["EDATE"])

# Reproject perimeter to match base grid
perimeter = perimeter.to_crs(basegrid.crs)

# 1.2 Process hotspots
modis_hotspots = pd.read_csv("modis_2021_Canada.csv")
viirs_hotspots = pd.read_csv("viirs-snpp_2021_Canada.csv")

viirs_hotspots.columns = modis_hotspots.columns
hotspots = pd.concat([modis_hotspots, viirs_hotspots])

# Add additional columns to hotspots
def decimal_day(acq_date, acq_time):
    jday = pd.to_datetime(acq_date).dayofyear
    time_decimal = int(acq_time) / 2400
    return jday + time_decimal

hotspots["JDAYDEC"] = hotspots.apply(lambda row: decimal_day(row["acq_date"], row["acq_time"]), axis=1)
hotspots["YEAR"] = pd.to_datetime(hotspots["acq_date"]).dt.year

# Filter hotspots by perimeter date
start_jday = perimeter["STARTDATE"].iloc[0].dayofyear
end_jday = perimeter["ENDDATE"].iloc[0].dayofyear
hotspots = hotspots[(hotspots["YEAR"] == perimeter["STARTDATE"].dt.year.iloc[0]) &
                    (hotspots["JDAYDEC"] >= start_jday - 30) &
                    (hotspots["JDAYDEC"] <= end_jday + 30)]

# Convert hotspots to GeoDataFrame
hotspots_gdf = gpd.GeoDataFrame(
    hotspots,
    geometry=gpd.points_from_xy(hotspots.longitude, hotspots.latitude),
    crs="EPSG:4326"
)
hotspots_gdf = hotspots_gdf.to_crs(basegrid.crs)

print('Buffering')

# Spatially crop hotspots to perimeter + 1000 m buffer
perimeter_buf = perimeter.buffer(1000)
hotspots_gdf = gpd.clip(hotspots_gdf, perimeter_buf)

# 2.0 Interpolation --------------------------------------------
# Prepare grid for interpolation
grid_fire = geometry_mask(
    [geom for geom in perimeter_buf.geometry],
    transform=basegrid.transform,
    invert=True,
    out_shape=(basegrid.height, basegrid.width)
)

print('Interpolating')
# Extract points for interpolation
# grid_pts = [(x, y) for y in range(grid_fire.shape[0]) for x in range(grid_fire.shape[1]) if grid_fire[y, x]]

rows, cols = np.where(grid_fire)
grid_pts = np.column_stack((cols, rows))

# Time zone correction (example assumes one time zone)
timezone_offset = -4 / 24  # Adjust as per the dataset
hotspots_gdf["JDAYDEC"] -= timezone_offset

# Kriging interpolation
coordinates = np.array(hotspots_gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
values = hotspots_gdf["JDAYDEC"].values
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# For now randomly sampling the data
sample_size = min(1000, len(coordinates))  # Adjust as needed
indices = np.random.choice(len(coordinates), sample_size, replace=False)
sampled_coords = coordinates[indices]
sampled_values = values[indices]
print('fitting')
gp.fit(sampled_coords, sampled_values)

#gp.fit(coordinates, values)

grid_coordinates = np.array(grid_pts)

print('predicting')
predicted, _ = gp.predict(grid_coordinates, return_std=True)

predicted_raster = np.full(grid_fire.shape, np.nan)  # Initialize raster with NaNs
predicted_raster[grid_fire] = predicted  # Assign predicted values to valid locations

# Assign interpolated values to the raster and save
#with rasterio.open("firearrival_decimal_krig.tif", "w", **basegrid.meta) as dst:
#    dst.write(predicted.reshape(grid_fire.shape), 1)

with rasterio.open("firearrival_decimal_krig.tif", "w", **basegrid.meta) as dst:
    dst.write(predicted_raster, 1)

print('Covariates')
# 3.0 Covariates ------------------------------------------------
# Static slope example
slope = rasterio.open("slope.tif")
# Reproject and extract slope to points (similar to above processing)
# ... Additional covariate handling ...

# 4.0 Summarize -----------------------------------------------------
# Load final points and summarize metrics
grid_pts_df = pd.read_csv("firespread_pts.csv")
grid_groups = grid_pts_df.groupby("DOB").agg({
    "fireday": "mean",
    "sprdistm": "mean",
    "firearea": "mean",
    "slope": "mean",
    "fwi": "mean"
}).reset_index()

# Save the summarized data
grid_groups.to_csv("firespread_groups.csv", index=False)

# Note: Use specialized fire modeling libraries for enhanced spread estimation.
