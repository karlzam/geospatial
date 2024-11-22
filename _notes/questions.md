# Questions

1. When referring to the pixel resolution of a satellite, we say "375m (VIIRS)" or "1000m (MODIS)". Is this actually
the algorithmically created pixel length/width? When looking into scan and track, the sizes of the scan/track vary,
and I'm assuming that its scan*track to get the area of the pixel. Or is this incorrect? So pixel resolution refers to 
the length or height of the pixel but not the area? 

2. Why are the GOES hotspot resolutions so different for very similar latitudes? 

3. Can I use the scan/track values to determine the pixel size or should I go the geometry route? 

4. The NBAC geometry is in "multipolygon" which has lots of little areas within a region corresponding to the fire - 
should I create a perimeter buffer first or use the hole-y perimeter?
