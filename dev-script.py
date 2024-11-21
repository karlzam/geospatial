import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth's surface.
    This function uses the haversine formula to compute the shortest distance
    over the Earth's surface between two points given their latitudes and longitudes.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point (in degrees)
    - lat2, lon2: Latitude and longitude of the second point (in degrees)

    Returns:
    - The great-circle distance between the two points (in kilometers)
    """

    R = 6371.0  # Earth's radius in kilometers (assumes a spherical Earth)

    # Convert latitude and longitude values from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate the difference in latitude and longitude between the two points
    dlat = lat2 - lat1  # Difference in latitude
    dlon = lon2 - lon1  # Difference in longitude

    # Apply the haversine formula:
    # a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
    # where:
    # - Δφ is the difference in latitude
    # - Δλ is the difference in longitude
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2

    # Compute the central angle (c) using the arctangent function
    # c = 2 * atan2(√a, √(1−a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the great-circle distance: d = R * c
    # Multiply the central angle (c) by the Earth's radius (R)
    distance = R * c

    return distance  # Return the distance in kilometers


def calculate_goes_pixel_resolution(lat, lon, satellite_longitude=-75.0, altitude=35786.0):
    """
    Determine the approximate resolution of a GOES pixel at a given latitude and longitude.

    Parameters:
    - lat: Latitude of the point (degrees)
    - lon: Longitude of the point (degrees)
    - satellite_longitude: Longitude of the GOES satellite (default is -75.0° for GOES-East)
    - altitude: Satellite altitude above Earth's surface (default 35,786 km)

    Returns:
    - Pixel resolution in kilometers
    """
    earth_radius = 6371.0  # Earth's radius in kilometers

    # Step 1: Compute sub-satellite point distance
    sub_sat_point_distance = haversine_distance(0, satellite_longitude, lat, lon)

    # Step 2: Compute slant range
    # Using the Pythagorean theorem in spherical coordinates
    slant_range = math.sqrt(altitude ** 2 + earth_radius ** 2 -
                            2 * altitude * earth_radius * math.cos(math.radians(sub_sat_point_distance / earth_radius)))

    # Step 3: Field of View to Ground Resolution
    fov = 17.4  # Field of view in degrees (assume full disk for ABI)
    resolution = slant_range * math.radians(fov) / 10800  # Convert FOV to radians and divide by ABI resolution grid size (10,800 pixels)

    return resolution


# Example usage
latitude = 30.0  # Latitude in degrees
longitude = -90.0  # Longitude in degrees

resolution = calculate_goes_pixel_resolution(latitude, longitude)
print(f"GOES pixel resolution at ({latitude}, {longitude}): {resolution:.2f} km")
