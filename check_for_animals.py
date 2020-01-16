# main.py

# Import packages
import numpy as np
import os
import mahotas as mh
import cv2
import itertools
import matplotlib as plt
import scipy.misc
from PIL import Image
import shutil
import re
import statistics as st


# Import functions
from Functions import create_dirs, check_sky, mask_seg, count_animals, smooth_animals, plot_image, get_centers_through_borders, get_bboxes, write_file, remove_bad_images

# 60m: minimum animal size = 20?

# Set parameters
sigma = 2
minimum_animal_size = 20
kernel = (5,5)
date = "2019-11/"
width = 512
height= 512
output_location = "../Data/labels/" + date #+ "/"


# Create the data directories
#create_dirs()



# Set all locations
input_location = "../Data/images/all/60m/"  #+ "/test/"
input_location_s = "../Data/semantic/all/60m/"  #+ "/test/"

# Run over semantic files and create ground truth train

for image_name in os.listdir(input_location_s): # + "/test"

    if image_name == "test":
        continue
        
        
    # Input image from directory
    print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
    input_image_s = input_location_s + image_name
    input_image = input_location + image_name
    
    if check_sky(input_image_s) == True:
            print("Warning, image contains sky. Removing from set.")
            #bad_image_list.append(image_name)
            continue
    
    # Mask everything but animals in image
    animals, mask = mask_seg(input_image_s)


    
    # Smooth animals with gaussian and remove tiny animals less than set size
    animals_smooth = smooth_animals(animals, sigma = sigma)

    
    # Count animals, if no animals are present, skip image
    labeled_animals, nr_objects = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel,plot=False)
    
    if nr_objects == 0:
        print("Zero animals in this picture", "\n")
        continue
    else:
        print("animals found")

        # Save animal images
        output_image = "../Data/only_animal_images/all/" + image_name 
        try:
            shutil.copyfile(input_image, output_image)
        except:
            pass