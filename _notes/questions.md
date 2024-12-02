# Questions

1. When referring to the pixel resolution of a satellite, we say "375m (VIIRS)" or "1000m (MODIS)". Is this actually
the algorithmically created pixel length/width? When looking into scan and track, the sizes of the scan/track vary,
and I'm assuming that its scan*track to get the area of the pixel. Or is this incorrect? So pixel resolution refers to 
the length or height of the pixel but not the area? 

2. Why are the GOES hotspot resolutions so different for very similar latitudes? 

3. Can I use the scan/track values to determine the pixel size or should I go the geometry route? 

4. The NBAC geometry is in "multipolygon" which has lots of little areas within a region corresponding to the fire - 
should I create a perimeter buffer first or use the hole-y perimeter?

5. In the GOES document, it says it also provides the current pixel size in km^2 for the fire product, but this isn't 
available through FIRMS. I have tried to use the haversine distance and altitude of the satellite, as well as using 
the scan and track values to determine pixel size. I can't find information on the FIRMS website defining what units the 
scan and track values are in - I found info for other sources that scan is the spatial resolution of the E/W direction 
of the scan, so I'm assuming it's in m's for the 4000+ values to make sense, but this is odd as the rest of the data
for all of the sources is in m's. But it can't be scan x track = surface area? Currently using just the largest track 
val

6. In an LSTM, it looks like there's always a sigmoid and tanh function within the 4 layers within each node. Does the 
keras version of the LSTM layer follow this? 