import math


def calculate_angular_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the angular distance between two points on a sphere.
    """
    print("\n--- Calculating Angular Distance ---")
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    print(f"Input lat/lon in radians: lat1={lat1:.6f}, lon1={lon1:.6f}, lat2={lat2:.6f}, lon2={lon2:.6f}")

    # Haversine formula for angular distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    angular_distance = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    print(f"Computed angular distance (radians): {angular_distance:.6f}")

    return angular_distance


def calculate_goes_pixel_resolution(lat, lon, satellite_longitude=-75.0, altitude=35786.0):
    """
    Determine the resolution of a GOES pixel at a given latitude and longitude.

    Parameters:
    - lat, lon: Latitude and longitude of the target point (degrees)
    - satellite_longitude: Longitude of the GOES satellite (default: -75.0° for GOES-East)
    - altitude: Satellite altitude above Earth's surface (default: 35,786 km)

    Returns:
    - Pixel resolution in kilometers
    """
    print("\n=== Calculating GOES Pixel Resolution ===")
    earth_radius = 6371.0  # Earth's radius in kilometers
    print(f"Earth radius: {earth_radius} km, Satellite altitude: {altitude} km")

    # Step 1: Calculate angular distance between the satellite's sub-point and the target point
    angular_distance = calculate_angular_distance(0, satellite_longitude, lat, lon)
    print(f"Angular distance from sub-satellite point: {angular_distance:.6f} radians")

    # Step 2: Compute the slant range (distance from satellite to point on Earth's surface)
    slant_range = math.sqrt(
        (earth_radius + altitude) ** 2
        - 2 * (earth_radius + altitude) * earth_radius * math.cos(angular_distance)
        + earth_radius ** 2
    )
    print(f"Computed slant range: {slant_range:.6f} km")

    # Step 3: Calculate angular resolution per pixel
    field_of_view = 17.4  # ABI field of view in degrees (full disk)
    angular_resolution_per_pixel = math.radians(field_of_view) / 10800  # Radians per pixel
    print(f"Angular resolution per pixel: {angular_resolution_per_pixel:.8f} radians")

    # Step 4: Calculate ground resolution
    ground_resolution = slant_range * angular_resolution_per_pixel  # Resolution in kilometers
    print(f"Computed ground resolution: {ground_resolution:.6f} km")

    return ground_resolution


# Example usage
latitude = 0.0  # Sub-satellite point (nominal resolution)
longitude = -75.0
resolution = calculate_goes_pixel_resolution(latitude, longitude)
print(f"\nFinal GOES pixel resolution at ({latitude}, {longitude}): {resolution:.6f} km")

# Test for another latitude (example: at 45 degrees latitude)
latitude = 45.0  # Latitude further from sub-satellite point
longitude = -75.0
resolution = calculate_goes_pixel_resolution(latitude, longitude)
print(f"\nFinal GOES pixel resolution at ({latitude}, {longitude}): {resolution:.6f} km")

# Test for a different longitude (example: at 90 degrees longitude)
latitude = 0.0
longitude = 90.0
resolution = calculate_goes_pixel_resolution(latitude, longitude)
print(f"\nFinal GOES pixel resolution at ({latitude}, {longitude}): {resolution:.6f} km")
