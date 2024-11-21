# General Notes 

## Satellites 
- [Spatial resolution, pixel size, and scale description](https://natural-resources.canada.ca/maps-tools-and-publications/satellite-imagery-elevation-data-and-air-photos/tutorial-fundamentals-remote-sensing/satellites-and-sensors/spatial-resolution-pixel-size-and-scale/9407) 


## NBAC

- Fire perimeters created for Canada
- Only hosts large fires in "forested areas"
- Buffering: the hotspot locations are in the center of the fire pixels, so in order to assess if hotspots are "inside
or outside the NBAC perimeter", we first need to adjust the perimeter to account for this resolution in hotspot locations

## FIRMS

[FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq)

### API Access

*   MODIS_NRT (2024-06-01 to 2024-09-10) - 1000m, 1-2 days
*   **MODIS_SP (2000-11-01 to 2024-05-31) - 1000m, 1-2 days**
*   **VIIRS_NOAA20_NRT (2019-12-04 to 2024-09-10) - 375m, 1 day**
*   VIIRS_NOAA21_NRT (2024-01-17 to 2024-09-10) - 375m
*   VIIRS_SNPP_NRT (2024-04-01 to 2024-09-10) - 375m
*   **VIIRS_SNPP_SP (2012-01-20 to 2024-03-31) - 375m, 1 day**
*   **LANDSAT_NRT (2022-06-20 to 2024-09-10) - 30m, 8 days (16 days each with 8 days out of phase)**
*   **GOES_NRT (2022-08-09 to 2024-09-10) - Geostationary **

### Attributes

- Only VIIRS and MODIS are very well described on FIRMS, the rest are somewhat explained in the attribute tables

- This page has descriptions of the tables that are available through FIRMS: 
https://firms.modaps.eosdis.nasa.gov/descriptions/


    # Hotspot attributes
    # lat/long: center of fire pixel, not necessarily the actual location of the fire
    # brightness: brightness temp 21/22 measured in kelvin
    # scan: spatial resolution of the E/W direction of the scan
    # track: spatial resolution of the N/S resolution of the scan
    # acq_date: date of aquisition
    # acq_time: time of overpass of satellite (in UTC)
    # TODO: Check if NBAC times are in UTC
    # confidence: intended to help users gauge the quality of individual hot spot/fire pixels
    # version: URT/RT/NRT, SP has no letters
    # brightness_T31: channel 31 brightness in Kelvin --> I don't actually see this in the dataframe?
    # FRP: fire radiative power (MW), pixel integrated
    # type (modis): 0 = vegetation, 1 = active volcano, 2 = other static source, 3 = offshore
    
    # What I see in the df
    # lat/long
    # bright_ti4 - VIIRS
    # scan - ALL
    # track - ALL
    # acq_date - ALL
    # acq_time - ALL
    # satellite - ALL EXCEPT GOES
    # instrument - ALL EXCEPT GOES
    # confidence - DIFFERENT FOR EVERY SATELLITE
    # version - DIFFERENT FOR EVERY SATELLITE
    # bright_ti5 - VIIRS
    # frp
    # daynight
    # type
    # geometry

#### VIIRS
- https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_VIIRS_Firehotspots.html
- https://www.earthdata.nasa.gov/data/tools/firms
- Scan/track in kilometers?

#### MODIS
- https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_MODIS_Firehotspots.html
- https://www.earthdata.nasa.gov/data/tools/firms
- Scan/track in kilometers? 

#### Landsat
- https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_Landsat_Firehotspots.html
- Track and scan is the pixel location either along-track (track) or across-track (scan) based on the OLI line of 
sight coordinate system
- Path and row refers to the WRS-2

#### GOES
- GOES algorithm document: https://www.star.nesdis.noaa.gov/goesr/product_land_fire.php
- Above document from this list: https://www.star.nesdis.noaa.gov/goesr/documentation_ATBDs.php
- Scan/track in meters? 



