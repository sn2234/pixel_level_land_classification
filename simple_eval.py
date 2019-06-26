
#%% Load dependencies

import os
os.environ['PROJ_LIB'] = r'C:\Users\nobody\Anaconda3\envs\cntk\Lib\site-packages\pyproj\data'

import numpy as np
import pandas as pd
import os, argparse, cntk, tifffile, warnings, osr
from osgeo import gdal
from gdalconst import *
from mpl_toolkits.basemap import Basemap
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from scipy.special import expit

#%% Load the model and evaluation data
# The default filename points to your model trained for one epoch.
# You can also try using our sample model, 250epochs.model
model_filename = r"D:\Prog\Projects\MachineLearning\Misc\Datasets\pixellevellandclassification\models\250epochs.model"
model = cntk.load_model(model_filename)

naip_filename = r"D:\Prog\Projects\MachineLearning\Misc\Datasets\pixellevellandclassification\evaluation_data\C14_NAIP.tif"
lc_filename = naip_filename.replace('_NAIP.tif', '_LandCover.tif')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    naip_image = np.transpose(tifffile.imread(naip_filename))  / 255.0
    true_lc_image = np.transpose(tifffile.imread(lc_filename))
true_lc_image[true_lc_image > 4] = 4

gdal.UseExceptions()

#%% Find the relevant subregion of the image
os.environ['PROJ_LIB'] = r'C:\Users\nobody\Anaconda3\envs\cntk\Library\share\proj'
def find_pixel_from_latlon(img_filename, lat, lon):
    ''' Find the indices for a point of interest '''
    img = gdal.Open(img_filename, GA_ReadOnly)
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

    xpos, ypos = world_map(lon, lat, inverse=False)
    xpos, ypos, _ = ct_to_img.TransformPoint(xpos, ypos)
    x = int((xpos - ulcrnrx) / xstep)
    y = int((ypos - ulcrnry) / ystep)

    return(x,y)

region_dim = 256
delta = int(region_dim / 2)
padding = 64

center_x, center_y = find_pixel_from_latlon(naip_filename, 37.055522, -78.638640)
true_lc_image = true_lc_image[center_x - delta:center_x + delta,
                              center_y - delta:center_y + delta].astype(np.float32)
naip_image = naip_image[:,
                        center_x - (delta + padding):center_x + delta + padding,
                        center_y - (delta + padding):center_y + delta + padding].astype(
                            np.float32)

#%% Applying the model across the region of interest
n_rows = int(region_dim / 128)

# The model's predictions will have five values for each x-y position:
# these can be used to find the relative predicted probabilities of
# each of the five labels.
pred_lc_image = np.zeros((5, true_lc_image.shape[0], true_lc_image.shape[1]))

for row_idx in range(n_rows):
    for col_idx in range(n_rows):
        # Extract a 256 x 256 region from the NAIP image, to feed into the model.
        sq_naip = naip_image[:,
                             row_idx * 128:(row_idx * 128) + 256,
                             col_idx * 128:(col_idx * 128) + 256]
        
        # Get the model's prediction for the center of that region
        sq_pred_lc = np.squeeze(model.eval({model.arguments[0]: [sq_naip]}))
        
        # Save the predictions in the appropriate region of the result matrix
        pred_lc_image[:,
                      row_idx * 128:(row_idx * 128) + 128,
                      col_idx * 128:(col_idx * 128) + 128] = sq_pred_lc

# You may disregard the RuntimeWarning about non-contiguous data

#%% Draw results

fig = plt.figure(figsize=(10, 40))
layers = pred_lc_image.shape[0]
total = layers+2

pred_lc_image_exp = expit(pred_lc_image)

threshold = pred_lc_image_exp.mean(axis=(1,2))

layerNames = [
    "No data",
    "Water",
    "Trees",
    "Herbaceous",
    "Barren/impervious"
]

def showImg(img, n, title):
    sub = fig.add_subplot(total, 2, n)
    sub.imshow(img)
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_title(title)

showImg(naip_image[:3].transpose((1,2,0)), 1, "Image")

for n in range(layers):
    start = 1+2*(n+total-layers)
    showImg(pred_lc_image_exp[n] > threshold[n], start, f"Prediction {layerNames[n]}")
    showImg(true_lc_image == n, start+1, f"Ground truth {layerNames[n]}")


#%% Visualization

color_map = np.asarray([[0,0,0],
                        [0,0,1],
                        [0,0.5,0],
                        [0.5,1,0.5],
                        [0.5,0.375,0.375]], dtype=np.float32)

def visualize_label_image(input_image, hard=True):
    num_labels, height, width = input_image.shape
    label_image = np.zeros((3, height, width))
    if hard:
        my_label_indices = input_image.argmax(axis=0)
        for label_idx in range(num_labels):
            for rgb_idx in range(3):
                label_image[rgb_idx, :, :] += (my_label_indices == label_idx) *\
                    color_map[label_idx, rgb_idx]
    else:
        input_image = np.exp(input_image) / np.sum(np.exp(input_image), axis=0)
        for label_idx in range(num_labels):
            for rgb_idx in range(3):
                label_image[rgb_idx, :, :] += input_image[label_idx, :, :] * \
                    color_map[label_idx, rgb_idx]
    label_image = np.transpose(label_image * 255).astype(np.uint8)
    return(label_image)

#%% Predicted land cover labels
img_labels_pred = visualize_label_image(pred_lc_image, hard=True)
plt.imshow(img_labels_pred)

#%% Ground-truth land cover labels
true_lc_labels = np.transpose(np.eye(5)[true_lc_image.astype(np.int32)], [2, 0, 1])
img_labels_true = visualize_label_image(true_lc_labels, hard=True)
plt.imshow(img_labels_true)

#%% Quantifying accuracy
fraction_correct = np.sum(true_lc_image == pred_lc_image.argmax(axis=0)) / region_dim**2
print('{:.1f}% of best-guess predictions were correct'.format(100 * fraction_correct))

pred_softmax = np.exp(pred_lc_image) / np.sum(np.exp(pred_lc_image), axis=0)
avg_prob_correct_label = np.sum(np.multiply(true_lc_labels, pred_softmax)) / region_dim**2
print('Average probability assigned to the true label: {:.1f}%'.format(
    100 * avg_prob_correct_label))
