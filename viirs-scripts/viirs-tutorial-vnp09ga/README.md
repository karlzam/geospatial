
# Working with Daily NASA VIIRS Surface Reflectance Data
---
# Objective:
This tutorial demonstrates how to work with the daily Suomi-NPP NASA VIIRS Surface Reflectance [(VNP09GA.001)](https://doi.org/10.5067/VIIRS/VNP09GA.001) data product.
The Land Processes Distributed Active Archive Center (LP DAAC) distributes the NASA-produced surface reflectance data products from the Visible Infrared Imaging Radiometer Suite (VIIRS) sensor aboard the joint NOAA/NASA Suomi-National Polar-orbiting Partnership (Suomi-NPP) satellite. The Suomi-NPP NASA VIIRS Surface Reflectance products are archived and distributed in the HDF-EOS5 file format. The Hierarchical Data Format version 5 (HDF5) is a NASA selected format of choice for standard science product archival and distribution and is the underlying format for HDF-EOS5. HDF-EOS is the standard format and I/O library for the Earth Observing System (EOS) Data and Information System (EOSDIS). HDF-EOS5 extends the capabilities of the HDF5 storage format, adding support for the various EOS data types (e.g., point, swath, grid) into the HDF5 framework.

In this tutorial, you will use Python to define the coordinate reference system (CRS) and export science dataset (SDS) layers as GeoTIFF files that can be loaded into a GIS and/or Remote Sensing software program.
***
### Example: Converting VNP09GA files into quality-filtered vegetation indices and examining the trend in vegetation greenness during July 2018 to observe drought in Europe.           
#### Data Used in the Example:  
- Data Product: VIIRS/NPP Surface Reflectance Daily L2G Global 1km and 500m SIN Grid Version 1 ([VNP09GA.001](http://dx.doi.org/10.5067/VIIRS/VNP09GA.001))  
     - Science Dataset (SDS) layers:  
        - SurfReflect_M3_1     
        - SurfReflect_M4_1    
        - SurfReflect_M5_1    
        - SurfReflect_M7_1   
---
# Topics Covered:
1. **Getting Started**    
    1a. Import Packages    
    1b. Set Up the Working Environment      
    1c. Retrieve Files    
2. **Importing and Interpreting Data**    
    2a. Open a VIIRS HDF5-EOS File and Read File Metadata     
    2b. Subset SDS Layers and read SDS Metadata   
3. **Generating an RGB Composite**    
    3a. Apply a Scale Factor  
    3b. Create an Image
4. **Quality Filtering**      
    4a. Decode Bit Values   
    4b. Apply Quality and Land/Water Masks   
5. **Functions**    
    5a. Defining Functions  
    5b. Executing Functions  
6. **Visualizing Results**      
    6a. Plot Quality-Filtered VIs       
    6b. Scatterplot Comparison of NDVI vs. EVI     
7. **Georeferencing**
8. **Exporting Results**    
    8a. Set Up a Dictionary  
    8b. Export Masked Arrays as GeoTIFFs  
9. **Working with Time Series**      
    9a. Automation of Steps 2-5, 7-8       
    9b. Time Series Analysis   
    9c. Time Series Visualization
---
# Prerequisites:
*A [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) account is required to download the data used in this tutorial. You can create an account at the link provided.*  
+ #### Python Version 3.6.1  
  + `h5py`  
  + `numpy`  
  + `matplotlib`  
  + `pandas`  
  + `datetime`  
  + `skimage`
  + `GDAL`
*To execute Sections 7-9, the [Geospatial Data Abstraction Library](http://www.gdal.org/) (GDAL) is required.*		
---
# Procedures:
#### 1. This tutorial uses the VNP09GA.001 tile h18v03 product on July 2, 2018. Use this link to download the file directly from the LP DAAC Data Pool:
 - https://e4ftl01.cr.usgs.gov/VIIRS/VNP09GA.001/2018.07.02/VNP09GA.A2018183.h18v03.001.2018184074906.h5.
#### 2. Section 9 features time series data, and you will need to download all of the files in order to execute it. All of the files used in Section 9 can be downloaded via either of following approaches:
 - [text file](https://git.earthdata.nasa.gov/projects/LPDUR/repos/nasa_viirs_surfacereflectance/browse/VIIRS_SR_Tutorial_DownloadLinks.txt) containing links to each of the files on the LP DAAC Data Pool.
 - [NASA Earthdata Search](https://search.earthdata.nasa.gov/search/granules?p=C1373412034-LPDAAC_ECS&m=39.9375!2.671875!2!1!0!0%2C2&tl=1518615645!4!!&qt=2018-07-01T00%3A00%3A00.000Z%2C2018-07-31T23%3A59%3A59.000Z&q=vnp09ga&ok=vnp09ga&sb=7.59375%2C51.328125%2C13.078125%2C57.234375) using the query in the link provided.
#### 3. Copy/clone/download the [NASA VIIRS Tutorial repo](https://git.earthdata.nasa.gov/rest/api/latest/projects/LPDUR/repos/nasa_viirs_surfacereflectance/archive?format=zip), or the desired tutorial from the LP DAAC Data User Resources Repository:
 - [Jupyter Notebook](https://git.earthdata.nasa.gov/projects/LPDUR/repos/nasa_viirs_surfacereflectance/browse/VIIRS_SR_Tutorial.ipynb)   

<div class="alert alert-block alert-warning" >
<b>NOTE:</b> This tutorial was developed for NASA VIIRS Daily Surface Reflectance HDF-EOS5 files and should only be used for the VNP09GA.001 product. </div>

## Python Environment Setup
> #### 1. It is recommended to use [Conda](https://conda.io/docs/), an environment manager, to set up a compatible Python environment. Download Conda for your OS [here](https://www.anaconda.com/download/). Once you have Conda installed, Follow the instructions below to successfully setup a Python environment on Windows, MacOS, or Linux.
> #### 2. Setup  
> - Using your preferred command line interface (command prompt, terminal, cmder, etc.) type the following to successfully create a compatible python environment:
>   - `conda create -n viirstutorial -c conda-forge --yes python=3.7 h5py numpy pandas matplotlib scikit-image gdal`   
>   - `conda activate viirstutorial`  
>   - `jupyter notebook`  

> If you do not have jupyter notebook installed, you may need to run:  
 > - `conda install jupyter notebook`  

  TIP: Having trouble activating your environment, or loading specific packages once you have activated your environment? Try the following:
  > Type: 'conda update conda'    

If you prefer to not install Conda, the same setup and dependencies can be achieved by using another package manager such as pip and the [requirements.txt file]() listed above.  
[Additional information](https://conda.io/docs/user-guide/tasks/manage-environments.html) on setting up and managing Conda environments.  
#### Still having trouble getting a compatible Python environment set up? Contact [LP DAAC User Services](https://lpdaac.usgs.gov/lpdaac-contact-us/).    

***
## Citations  

- Krehbiel, C., (2018). Working with Daily NASA VIIRS Surface Reflectance Data [Jupyter Notebook]. NASA EOSDIS Land Processes Distributed Active Archive Center (LP DAAC), USGS/Earth Resources Observation and Science (EROS) Center, Sioux Falls, South Dakota, USA. Accessed Month, DD, YYYY. https://git.earthdata.nasa.gov/projects/LPDUR/repos/nasa-viirs-tutorial/browse


- Vermote, E., Franch, B., Claverie, M. (2016). VIIRS/NPP Surface Reflectance Daily L2G Global 1km and 500m SIN Grid V001 [Data set]. NASA EOSDIS Land Processes DAAC. doi: 10.5067/VIIRS/VNP09GA.001

***
<div class="alert alert-block alert-info">
<h1> Contact Information </h1>   
    
<b>Contact:</b> LPDAAC@usgs.gov  
    
<b>Voice:</b> +1-866-573-3222     
    
<b>Organization:</b> Land Processes Distributed Active Archive Center (LP DAAC)    
    
<b>Website:</b> https://lpdaac.usgs.gov/  
    
<b>Date last modified:</b> 02-23-2022  

Work performed under USGS contract G15PD00467 for LP DAAC.

LP DAAC Work performed under NASA contract NNG14HH33I. 
