import math
import pandas as pd

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371

def calculate_cell_size(scan, track, latitude):
    """
    Calculate the physical cell size for a given scan and track value.

    Parameters:
        scan (float): GOES scan value in radians (east-west).
        track (float): GOES track value in radians (north-south).
        latitude (float): Latitude in degrees.

    Returns:
        tuple: (cell_width_km, cell_height_km)
    """
    # Convert latitude to radians
    latitude_rad = math.radians(latitude)

    # Calculate cell dimensions
    cell_height_km = track * EARTH_RADIUS_KM  # North-south distance
    cell_width_km = scan * EARTH_RADIUS_KM * math.cos(latitude_rad)  # East-west distance adjusted by latitude

    return cell_width_km, cell_height_km

# Example FIRMS data
# Get scan and track vals in radians (using arc length = radius x angle in radians, or s=rtheta)
e_radius = 6371000 # in m
data = {
    "latitude": [50],  # Latitudes in degrees
    # 451, 3261 (scan, track)
    "scan": [451/e_radius],  # Scan values in radians
    "track": [3261/e_radius]  # Track values in radians
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Calculate cell sizes for each hotspot
df["cell_width_km"], df["cell_height_km"] = zip(*df.apply(
    lambda row: calculate_cell_size(row["scan"], row["track"], row["latitude"]), axis=1
))

# Output results
print(df)

# Save to CSV (optional)
#df.to_csv("goes_cell_sizes.csv", index=False)
