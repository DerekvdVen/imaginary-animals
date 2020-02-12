# check_for_animals.py

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
import argparse

# Import functions
from Functions import create_dirs, check_sky, mask_seg, check_for_animals, count_animals, smooth_animals, plot_image, get_centers_through_borders, get_bboxes, write_file, remove_bad_images

parser = argparse.ArgumentParser(description='check_for_animals')
parser.add_argument('-dir', default="30m", type=str, help='directory named after distance: 30m, 60m, 3060m')
args = parser.parse_args()
print(args)
dir = args.dir
print(dir)

# 60m: minimum animal size = 20?

# Set parameters
sigma = 2
width = height = 512

#dir = "30m"

# Create the data directories
#create_dirs()



# Set all locations
input_location = "../Data/images/all/" + dir + '/'  #+ "/test/"
input_location_s = "../Data/semantic/all/" + dir + '/' #+ "/test/"

# Run over semantic files and create ground truth train

for image_name in os.listdir(input_location_s): # + "/test"

    if image_name == "test":
        continue
        
    # Input image from directory
    print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
    input_image_s = input_location_s + image_name
    input_image = input_location + image_name
    
    # Mask everything but animals in image
    animals, mask = mask_seg(input_image_s)

    animal_count = check_for_animals(animals)
    if animal_count != 0:
        output_image = "../Data/only_animal_images/all/" + dir + '/' + image_name 
        try:
            shutil.copyfile(input_image, output_image)
        except:
            pass
