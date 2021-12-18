# -*- coding: utf-8 -*-
"""
CP8307 Python Scripts for Performing Segmentation on the Kaggle fish dataset.
All relevant papers / tutorials will be referenced alongside the code that
uses them.

Written by: Christopher Kolios
Group Member: Yiannis Varnava

Tutorials and Resources Used:

Fish Dataset: https://www.kaggle.com/crowww/a-large-scale-fish-dataset

Scikit image documentation on thresholding:
https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

https://scikit-image.org/docs/dev/auto_examples/ (See detection of features and objects)

https://en.wikipedia.org/wiki/Thresholding_(image_processing)
"""

# Required imports include...
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage import io # For reading in / writing images
from skimage import feature
from skimage.filters import try_all_threshold
from skimage.filters import gaussian
from skimage.viewer import ImageViewer # For viewing images
from skimage.color import rgb2gray # To convert RGB to gray
from skimage.filters import threshold_otsu, threshold_local
from skimage.segmentation import chan_vese
from skimage import segmentation, color
from skimage.future import graph
from skimage import filters, morphology
from skimage.segmentation import flood, flood_fill
from skimage.filters import median

from skimage.morphology import erosion
from skimage.morphology import area_opening
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

import statistics
from mpl_toolkits.axes_grid1 import ImageGrid

import os
import sys

import numpy as np

def image_mean(input_image):
	# Will just use numpy as image mean is trivial
	return np.mean(input_image)

def image_thresholding(in_image, automatic=True):
	'''
	Thresholds a list of images

	Parameters
	----------
	list_of_images : list of np arrays
		The images to threshold.
	automatic : Boolean, optional
		Whether or not to use automatic thresholding. The default is True.

	Returns
	-------
	tempbinary.

	'''
	#for image in list_of_images:
	image = in_image
	threshold_difference = 100
	new_threshold = 100
	while threshold_difference > 0.005: # This is custom
		
		if (threshold_difference == 100): # i.e. first iteration
			curr_im_mean = image_mean(image) # Select initial threshold value (image mean)
		else:
			curr_im_mean = new_threshold

		# Divide the image into two portions
		temp_binary = np.where(image > curr_im_mean, 1, image)
		temp_binary = np.where(image <= curr_im_mean, 0, temp_binary)
		
		# Find the average of the two new images
		foreground_mean = image_mean(image[temp_binary == 1])
		background_mean = image_mean(image[temp_binary == 0])
		
		# New threshold = average of the two means
		new_threshold = (foreground_mean + background_mean)/2
		
		if automatic:
			# Difference between old and new thresholds
			threshold_difference = abs(new_threshold - curr_im_mean)
		else:
			threshold_difference = -1 # if want to see 1 iteration
	
	return temp_binary

# Based on notebook from Advanced Computer Vision with TensorFlow (copied)
def compute_iou(y_true, y_pred, i=1):
  # value of 1 corresponds to the fish. 0 corresponds to the background.
  eps = 0.00001 # for smoothing
  intersection = np.sum((y_pred == i) * (y_true == i)) # AND operation to find INTERSECTION
  y_true_area = np.sum((y_true == i))
  y_pred_area = np.sum((y_pred == i))
  union = y_true_area + y_pred_area # OR operation for union
  iou = (intersection + eps) / (union - intersection + eps) # subtract intersection to remove double count
  return round(iou,2)

# Based on notebook from Advanced Computer Vision with TensorFlow
def compute_dice_score(y_true, y_pred, i=1):
  # value of 1 corresponds to the class of the fish. 0 corresponds to the background.
  eps = 0.00001  
  intersection = np.sum((y_pred == i) * (y_true == i))
  y_true_area = np.sum((y_true == i))
  y_pred_area = np.sum((y_pred == i))
  union = y_true_area + y_pred_area
  dice_score =  2 * ((intersection + eps) / (union + eps))
  return round(dice_score,2)

def get_coord(imline, direction): # For the RAG boundary sweeping (custom)
	#print("imline:", len(imline))
	coord = 0
	for pixel in imline:
		coord += 1
		if (pixel.all() != 0):
			break
	if (coord == len(imline)):
		coord -= 1
	if (direction == "reverse"):
		return len(imline) - coord
	
	return coord

bg_colour = (0.1059, 0.1294, 0.1725, 1.0000)
cmap = matplotlib.colors.ListedColormap(['black', 'white'])
def visualize(display_list): # For displaying the images and formatting
	fig = plt.figure(figsize=(10, 10))
	#title = ['Predicted Mask', 'Predicted Mask', 'Predicted Mask', 'Predicted Mask']
	#for i in range(len(display_list)):
	#	plt.subplot(1, len(display_list), i+1)
	#	plt.title(title[i], color='white')
	#	plt.imshow(display_list[i], cmap=cmap)
	#	plt.axis('off')
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 3),  # creates 2x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )
	for ax, im in zip(grid, display_list):
		ax.set_axis_off()
		ax.set_title("Predicted Mask", color='white')
		ax.imshow(im, cmap=cmap)
        
	fig.patch.set_facecolor(bg_colour)
	plt.axis('off')
	
	plt.savefig("test.png", bbox_inches='tight')
	plt.show()
	

if __name__ == "__main__":
	print("Initializing Program")
	
	'''
	Here is some quick Python code to iterate over all files in a the fish dataset
	Note: if we want the NA version just change to NA_Fish_Dataset (use augmented)
	'''
	# Test images: 1246, 282, 1306, 1550
	base_dir = r".\test_images"
	roi_dir = r".\test_masks"
	image_list = []
	colour_image_list = []
	roi_list = []
	for root, dirs, files in os.walk(base_dir):
		#fishtype = root.rsplit("\\")[-1]
		#print(root, dirs, files)
		#if ("GT" in root): # Ground Truth files
			#continue
			## print files
		#else: # Input files
			#if (len(files) > 0): # i.e. if a dir:
				#if ("Black Sea Sprat" in root): # TEMP: For testing
					##print(root)
					#print(fishtype)
					##print(files)
					#for file in files[48:49]: # TEMP: For testing
						#print(file)
						#image_list.append(io.imread(root + "\\" + file, as_gray=True))
						#colour_image_list.append(io.imread(root + "\\" + file, as_gray=False))
		#for file in files[132:133]: # Or is it 30:31?
			#print(file)
			#if (file ==  "1550_00039.png"): # 1246_00004.png, 282_00022.png, 1306_00030.png, 1550_00039.png
				#print(file)
			#print(file)
			#image_list.append(io.imread(root + "\\" + file, as_gray=True))
			#colour_image_list.append(io.imread(root + "\\" + file, as_gray=False))
			#roi_list.append(io.imread(roi_dir + "\\" + file, as_gray=True))
			#if (file == "1246_00004.png" or file == "282_00022.png" or file == "1306_00030.png" or file == "1550_00039.png"):
		for file in ["1246_00004.png", "282_00022.png", "1306_00030.png", "1550_00039.png"]:
				image_list.append(io.imread(root + "\\" + file, as_gray=True))
				colour_image_list.append(io.imread(root + "\\" + file, as_gray=False))
				roi_list.append(io.imread(roi_dir + "\\" + file, as_gray=True))
			
		
	#for image in image_list: # To show all plots
		#io.imshow(image)
		#plt.show()
	
	''' 
	First let's try image thresholding as a segmentation method.
	Reference: https://en.wikipedia.org/wiki/Thresholding_(image_processing)
	
	Image thresholding is pretty simple at its core - we just generate binary
	images such that pixels are black if they are not part of the "foreground"
	and they are white if they are. In our case the "foreground" is the fish,
	and the background is its background.
	
	Below, we implement automatic thresholding, based on some input threshold.
	We will keep thresholding while the difference between the foreground
	and background means (a.k.a. the new threhsold) minus the old threshold
	is greater than some factor (here: 0.005).
	
	''' 
	threshold_images = []
	
	for image in image_list:
		thresh_im = image_thresholding(image)
		thresh_im = np.array(thresh_im)
# 		
# 		groundtruth = roi_list[0]
# 		groundtruth[groundtruth != 0] = 1
# 		
# 		print("IoU", compute_iou(groundtruth, thresh_im)) # Just a test
# 		print("Dice", compute_dice_score(groundtruth, thresh_im))
		
		#io.imshow(thresh_im)
		threshold_images.append(thresh_im)
		
	
	'''
	We can see that in the case of 1 iteration, while the body of the fish
	is indeed separated from the surrounding region, there is plenty of the
	background that is highlighted as white.
	
	When we perform automatic thresholding, almost the entire board ends up
	turning white, but we notice a better (in some ways) border surrounding
	our fish.
	'''
		
	'''
	Next, let's test the effects of image smoothing on the above procedure.
	We notice some jagged edges, and would like to see if standard Gaussian
	smoothing will help with this.
	'''
	
# 	gauss_images = []
# 	for image in image_list:
# 		gauss_images.append(gaussian(image, 2)) # Second arg is sigma
# 		
# 	thresh_im = image_thresholding(gauss_images, roi_list)
# 	groundtruth = roi_list[0]
# 	groundtruth[groundtruth != 0] = 1
# 	
# 	print("IoU", compute_iou(groundtruth, thresh_im)) # Just a test
# 	print("Dice", compute_dice_score(groundtruth, thresh_im))
# 	sys.exit()
	
	'''
	We see that Gaussian filtering helps reduce the amount of speckled noise
	that is present in the images.
	'''
	
	'''
	Unfortunately even with Gaussian filtering we still have the problem of
	a lot of the background pixels being white, and some of the fish pixels
	being black. Even though thresholding is powerful, we run into limitations
	due to the variety of image contrast levels both on and off the fish.
	'''
	
	'''
	Next, let's try using edge detection to try and get a clear boundary for 
	each fish.
	'''
	
	#edges1 = feature.canny(image_list[0])
	#io.imshow(edges1)
	#edges3 = feature.canny(image_list[0], 2.5)
	#io.imshow(edges3)
	
	'''
	Using Canny edge detection, we get a good representation of the boundary
	of the image, as well as that of the fish itself.
	'''
	
	'''
	However, the problem is that even with the boundary it is hard to know
	exactly where the fish is. We don't have a human to fill in any border
	gaps or determine where to start a seed to grow
	'''
	
	'''
	Consequently, we will use some segmentation methods that do not rely on
	human intervention, starting with Chan-Vese...
	'''
	CVimages = []
	
	for image in image_list:
		cv = chan_vese(image, mu=2, lambda1=1, lambda2=1, tol=1e-3, #2.5, 1, 1.05
	               max_iter=200, dt=0.5, init_level_set="checkerboard",
	               extended_output=True)
		#med = median(cv[0], square(3))
		#gauss = gaussian(cv[0])
		erod = erosion(cv[0]) # Then we perform erosion to fill in background holes
		erod = erosion(erod)
		arc = area_opening(erod)
		#arc = area_opening(erod, area_threshold=128)
		
		#io.imshow(cv[0])
		#io.imshow(cv[0])
		#io.imshow(arc)
		CVimages.append(arc)
		
# 		
# 		groundtruth = roi_list[0]
# 		groundtruth[groundtruth != 0] = 1
# 		
# 		print("IoU", compute_iou(groundtruth, cv[0])) # Just a test
# 		print("Dice", compute_dice_score(groundtruth, cv[0]))
# 		
# 		print("aIoU", compute_iou(groundtruth, arc)) # Just a test
# 		print("aDice", compute_dice_score(groundtruth, arc))
# 		
# 		#io.imshow(cv[0])
# 		#io.imshow(med)
# 		sys.exit()
	
	
	'''
	Next let's try RAG's.
	
	From:
	https://scikit-image.org/docs/dev/auto_examples/segmentation/
	plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-
	plot-rag-mean-color-py
	'''
	RAGims = []
	allIOUs = []
	allDICE = []
	counter = 0
	for img in colour_image_list:
		labels1 = segmentation.slic(img, compactness=30, n_segments=300, start_label=1) #was 30, 400, 1
		out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
		
		g = graph.rag_mean_color(img, labels1)
		labels2 = graph.cut_threshold(labels1, g, 45) # was 29
		out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0) # 300, 45?
		
		# Convert to gray for flood fill
		gray_version = rgb2gray(out2)
		
		#io.imshow(out2)
		#io.imsave("out1ColourSuperPixels.png", out1)
		#io.imsave("out2ColourRAG.png", out2)
		#io.imsave("grayversion.png", gray_version)
		#io.imshow(gray_version)
		
		# Iterate from top left to top right until found first non-black pixel
		coords = []
		coords.append((0, get_coord(gray_version[0], "normal"))) # Top left to right (y, x)
		coords.append((0, get_coord(gray_version[0][::-1], "reverse"))) # Top right to left
		coords.append((len(gray_version)-1, get_coord(gray_version[len(gray_version)-1], "normal"))) # Bottom left to right
		coords.append((len(gray_version)-1, get_coord(gray_version[len(gray_version)-1][::-1], "reverse"))) # Bottom right to left
		coords.append((get_coord(gray_version[:,0], "normal"), 0)) # Top left to bottom
		coords.append((get_coord(gray_version[:,0][::-1], "reverse"), 0)) # Bottom left to top
		coords.append((get_coord(gray_version[:,len(gray_version[0])-1], "normal"), len(gray_version[0])-1)) # Top right to bottom
		coords.append((get_coord(gray_version[:,len(gray_version[0])-1][::-1], "reverse"), len(gray_version[0])-1)) # Top right to bottom # Bottom right to top
		
		# Paint the image from the first non-black pixel
		filledim = gray_version
		for coord in coords:
 			#print(coord)
 			filledim = flood_fill(filledim, (coord), 0)
		
		# Turn into a mask
		filledim[filledim != 0] = 1
		
		RAGims.append(filledim)
		#io.imsave("finalResult.png", filledim)
		#io.imshow(filledim)
		
		# Test groundtruth
		#groundtruth = io.imread(r"C:\Users\ckoli\Desktop\FishData\Fish_Dataset\Fish_Dataset\Black Sea Sprat\Black Sea Sprat GT\00049.png", as_gray=True)
		groundtruth = roi_list[counter]
		groundtruth[groundtruth != 0] = 1
		
		#io.imshow_collection([filledim, groundtruth])
		
		#print(filledim.max(), groundtruth.max())
		
		allIOUs.append(compute_iou(groundtruth, filledim))
		allDICE.append(compute_dice_score(groundtruth, filledim))
		
		print(counter)
		print("IoU", allIOUs[counter]) # Just a test
		print("Dice", allDICE[counter])
		
		counter += 1
	
	# Get average and median of IoUs
	iou_avg_val = sum(allIOUs)/len(allIOUs)
	iou_med = statistics.median(allIOUs)
	print("IoU avg:", iou_avg_val, "IOU med:", iou_med)
	# Get average and median of DICE
	dice_avg_val = sum(allDICE)/len(allDICE)
	dice_med = statistics.median(allDICE)
	print("DICE avg:", dice_avg_val, "DICE med:", dice_med)

	#print(RAGims)
	totalIms = []
	for i in range(4):
		for j in range(3):
			if (j == 0): # RAG
				totalIms.append(RAGims[i])
			elif (j == 1): #CV
				totalIms.append(CVimages[i])
			else:
				totalIms.append(threshold_images[i])
		
	#visualize(totalIms)
	
	i = 0 # Saves all images
	for im in totalIms:
		save_string = str(i)+".png"
		io.imsave(save_string, im)
		i += 1