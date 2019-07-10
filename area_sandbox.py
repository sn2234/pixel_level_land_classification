
#%% Load dependencies

import os
os.environ['PROJ_LIB'] = r'C:\Users\nobody\Anaconda3\envs\cntk\Lib\site-packages\pyproj\data'

import numpy as np
import pandas as pd
from osgeo import gdal
from gdalconst import *
from mpl_toolkits.basemap import Basemap
import os, argparse, cntk, tifffile, warnings, osr
from collections import namedtuple
import matplotlib.pyplot as plt

#%% Definitions
naip_filename = r"D:\Prog\Projects\MachineLearning\Misc\Datasets\pixellevellandclassification\evaluation_data\C14_NAIP.tif"
lc_filename = naip_filename.replace('_NAIP.tif', '_LandCover.tif')

gdal.UseExceptions()
os.environ['PROJ_LIB'] = r'C:\Users\nobody\Anaconda3\envs\cntk\Library\share\proj'

#%% Let's try
img = gdal.Open(naip_filename, GA_ReadOnly)
img_proj = osr.SpatialReference()
img_proj.ImportFromWkt(img.GetProjection())
ulcrnrx, xstep, _, ulcrnry, _, ystep = img.GetGeoTransform()

world_map = Basemap(lat_0=0,
                    lon_0=0,
                    llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180,
                    resolution='c', projection='stere')
world_proj = osr.SpatialReference()
world_proj.ImportFromProj4(world_map.proj4string)
ct_to_img = osr.CoordinateTransformation(world_proj, img_proj)

transform  = img.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = transform[5]

(xOrigin, yOrigin, pixelWidth, pixelHeight)