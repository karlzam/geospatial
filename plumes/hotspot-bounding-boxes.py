
from shapely.geometry import Point, box
import math
import matplotlib.pyplot as plt
import geopandas as gpd

def create_bounding_box(center_point, km):
    # Convert the point to (latitude, longitude)
    lat, lon = center_point.y, center_point.x

    # Earth radius in kilometers
    earth_radius_km = 6371

    # Calculate the latitude and longitude degree offsets for 20 km
    lat_offset = km / earth_radius_km * (180 / math.pi)  # Degree offset for latitude
    lon_offset = km / (earth_radius_km * math.cos(math.radians(lat))) * (180 / math.pi)  # Degree offset for longitude

    # Define the corners of the bounding box
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset

    # Create the bounding box
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)

    return bounding_box

hotspots = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\all-hotspots-oob-2023.shp')

bounding_boxes = []
for idx, row in hotspots.iterrows():
    bounding_box = create_bounding_box(row['geometry'], 2)
    bounding_boxes.append(bounding_box)

hotspots['bbox'] = bounding_boxes

# Create a new column 'bbox_coords' that contains the exterior coordinates of the bounding boxes
hotspots = hotspots.assign(bbox_coords=hotspots['bbox'].apply(lambda geom: geom.bounds))

# If you want the bounding box coordinates as a list of tuples
hotspots['bbox_coords'] = hotspots['bbox'].apply(lambda geom: list(geom.exterior.coords))

hotspots['coord-0'] = hotspots['bbox_coords'].apply(lambda coords: coords[0][0] if coords else None)
hotspots['coord-2'] = hotspots['bbox_coords'].apply(lambda coords: coords[2][0] if coords else None)

hotspots['coord-1'] = hotspots['bbox_coords'].apply(lambda coords: coords[0][1] if coords else None)
hotspots['coord-3'] = hotspots['bbox_coords'].apply(lambda coords: coords[2][1] if coords else None)

print('test')

hotspots.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\hotspots-w-bbox.shp')
hotspots.to_excel(r'C:\Users\kzammit\Documents\Sept23-Fire\hotspots-w-bbox.xlsx')

# NBAC = NBAC.assign(start_dt = lambda x: pd.to_datetime(x['start_date'], format=date_format))

print('test')