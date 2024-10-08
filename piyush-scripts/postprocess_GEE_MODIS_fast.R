# try to process data returned by GEE

library(terra)
library(raster)
library(rnaturalearth)
library(sf)
library(tmap)
library(exactextractr)
library(lubridate)
library(stringr)
`%ni%` <- Negate(`%in%`)

library(future)
library(future.apply)

#plan(multisession(workers = 4))
plan(multisession, workers = 4)
options(future.globals.maxSize=2000*1024^2)

#sq_ls <- future_lapply(1:1000, function(x) x^2)


# set paths to data
# path.test = "C:/Piyush_Test/MCD64A1_burndate_2021-0000000000-0000013568.tif"
# path = "C:/Piyush_Test/"
# path.era5.lsm = "C:/Piyush_Test/ERA5_landsea_mask.nc"
# #path.ppa.grid = "/home/piyush/PROJECTS/2021 Fire Season Analysis/Data/Grid_Aseem/Sample_grid_dly_z.grd"

# set paths to data - MACBOOK
path.test = "/Users/piyush/RESEARCH/PROJECTS/2021 Fire Season Analysis/Data/MODIS_AB_GEE_nominalscale/MCD64A1_burndate_2017-0000000000-0000013568.tif"
path = "/Users/piyush/RESEARCH/PROJECTS/2021 Fire Season Analysis/Data/MODIS_AB_GEE_nominalscale/"
path.era5.lsm = "/Users/piyush/RESEARCH/PROJECTS/2021 Fire Season Analysis/Data/ERA5_landsea_mask.nc"

# set paths to data - DESKTOP
#path.test = "/home/piyush/PROJECTS/2021 Fire Season Analysis/Data/MODIS_AB_GEE_nominalscale/MCD64A1_burndate_2017-0000000000-0000013568.tif"
#path = "/home/piyush/PROJECTS/2021 Fire Season Analysis/Data/MODIS_AB_GEE_nominalscale/"
#path.era5.lsm = "/home/piyush/PROJECTS/2021 Fire Season Analysis/Data/ERA5_landsea_mask.nc"


#setwd("/Users/piyush/RESEARCH/PROJECTS/2021 Fire Season Analysis/Data/")
setwd("/home/piyush/PROJECTS/2021 Fire Season Analysis/Data/")
#setwd("C:/Piyush_Test/")

# set parameters
year = 2021
resolution = 1

# load shape files for extracting to North America

sf_use_s2(FALSE) # fix required.
world <- ne_countries(scale = "medium", returnclass = "sf")
#world <- ne_states(returnclass="sf")
northamerica = world %>% subset(admin %in% c("United States of America", "Canada"))
canada = world %>% subset(admin %in% c("Canada"))
crop_ext = extent(northamerica)
crop_ext[2] = extent(canada)[2]
northamerica = st_crop(northamerica, crop_ext)
northamerica = northamerica %>% st_buffer(.00001) %>% st_union()
#northamerica = st_union(northamerica)
tm_shape(northamerica) + tm_polygons()

# load test - need this later for CRS
r2 = rast(path.test)

#############################################################################################
# PREPARE THE GRIDS!

# load land sea mask for ERA5
era5.grid.lsm = rast(path.era5.lsm)
era5.grid.lsm = terra::rotate(era5.grid.lsm)
era5.grid.lsm <- crop(era5.grid.lsm, northamerica)
crs(era5.grid.lsm) <- NULL
crs(era5.grid.lsm) <- "epsg:4326"
ext(era5.grid.lsm)

# three different grids at different resolutions
era5.1deg = terra::aggregate(era5.grid.lsm, 4, fun=mean)
#era5.1deg = crop(era5.1deg, extent(as_Spatial(northamerica)))
tm_shape(era5.1deg) + tm_raster() + tm_shape(northamerica) + tm_borders()

era5.0p5deg = aggregate(era5.grid.lsm, 2, fun=mean)
tm_shape(era5.0p5deg) + tm_raster() + tm_shape(northamerica) + tm_borders()

era5.0p25deg = era5.grid.lsm
tm_shape(era5.0p25deg) + tm_raster() + tm_shape(northamerica) + tm_borders()

# fix the grid here to just one and continue

#era5.grid = era5.1deg
if (resolution == 1) {
  era5.grid = era5.1deg
} else if (resolution == 0.5) {
  era5.grid = era5.0p5deg
} else if (resolution == 0.25) {
  era5.grid = era5.0p25deg
} else {
  stop("warning: no grid found at that resolution")
}

# start the regridding procedure.
era5.poly = as.polygons(era5.grid,dissolve = F)
# project to native projection of MODIS data (sinusoidal grid)
era5.poly.sf = st_as_sf(era5.poly)
era5.poly.sf.sinu = st_transform(era5.poly.sf, crs(r2))
era5.poly.sf.sinu.land = era5.poly.sf.sinu[era5.poly.sf.sinu$lsm>0.,]

era5.grid <- crop(era5.grid,era5.poly)

###########################################################################
# main code for extracting. 
###########################################################################

# preload MODIS tiles

modis.list <- list.files(path=path, full.names=T, pattern=paste0('MCD64A1_burndate.*',year,'.*\\.tif'))
# filter out any tiles with no data - will lead to speedups with exact_extract
modis.include = (1:length(modis.list))*FALSE
for (i in 1:length(modis.list)) {
  test = vrt(modis.list[i])
  test = which(terra::values(test) > 0)
  if (length(test) > 0) {
    modis.include[i] = i
  }
  rm(test)
}
modis = vrt(modis.list[modis.include])

# initialize output
ABdoy.regrid.year =  rast(replicate(365, era5.grid))
ABdoy.regrid.year[] = 0
names(ABdoy.regrid.year) <- format(seq.Date(as.Date(paste0(year,"-01-01")),as.Date(paste0(year,"-12-31")),by = "day"),"%b-%d")

# loop through months
for (m in 1:12) {
  print(paste0("Processing month: ", m))
  # calculate number of days for this month and year
  month.date <- as.Date(paste0(year, "-", m, "-1"))
  numdays <- days_in_month(month.date)[[1]]
  #j_days <- as.numeric(format(seq.Date(as.Date(paste0(year,"-0",m,"-01")),as.Date(paste0(year,"-0",m,"-01"))+numdays-1,by="days"),"%j"))
  doy.dates = seq.Date(as.Date(paste0(year,"-",m,"-01")),as.Date(paste0(year,"-",m,"-01"))+numdays-1,by="days")
  doy.list = yday(doy.dates)
  
  r <- modis[[m]]
  res <- terra::res(r)
  
  # create empty 
  dailyAreaBurned <- r
  values(dailyAreaBurned) <- 0
  
  mask.ab <- r
  mask.ab[] <- 0
  #init(dailyAreaBurned, fun=0)
  
  # get actual values since faster than dealing with original raster
  r.val <- terra::values(r)
  
  #dailyAreaBurned = 
  
  # I think every day has some area burned so this doesn't really help - comment
  #out <- which(doy.list %ni% unique(r.val))
  #if(length(out) > 0)  j_days <- j_days[-out]
  
  out = future_lapply(doy.list, function(doy) {
    test = which(r.val == doy)
    
    if(length(test) > 0){
      
      #doy.date <- as.Date(paste0(year, "-", m, "-", dom))
      #doy <- yday(doy.date)
      print(paste0("Processing doy: ", doy))
      # extract to target polygons
      dailyAreaBurned = mask.ab
      dailyAreaBurned[test] <- 1
      exact_extract(x = dailyAreaBurned, 
                                                     y = era5.poly.sf.sinu, 
                                                     fun = 'sum', 
                                                     weights='area') * res[1] * res[2] / 1e4 
    } else {
      rep(0, dim(ABdoy.regrid.year)[1]*dim(ABdoy.regrid.year)[2])
    }
    
  })
  
  for (doy in doy.list) {
    #timer <- Sys.time()
    test = which(r.val == doy)
    
    if(length(test) > 0){
      
      #doy.date <- as.Date(paste0(year, "-", m, "-", dom))
      #doy <- yday(doy.date)
      print(paste0("Processing doy: ", doy))
      # extract to target polygons
      dailyAreaBurned = mask.ab
      dailyAreaBurned[test] <- 1
      era5.poly.sf.sinu$area_burned <- exact_extract(x = dailyAreaBurned, 
                                                     y = era5.poly.sf.sinu, 
                                                     fun = 'sum', 
                                                     weights='area') * res[1] * res[2] / 1e4 
            ABdoy.regrid.year[[doy]][] = era5.poly.sf.sinu$area_burned
    

    }
    #print("Daily Timer: ")
    #print(Sys.time() - timer)
  }
  #print("Monthly Timer: ")
  #print(Sys.time()-timer)
   
}

################################################################################
# export netcdf
outfile = paste0("MODIS_AB_NA_res", str_replace(as.character(resolution), "[.]", "p"), "_", year, ".nc")
writeRaster(raster(ABdoy.regrid.year), outfile, overwrite=TRUE, format="CDF", varname="AB", varunit="ha", 
            longname="area burned - ha", xname="lon", yname="lat")
