# main.py

# Import packages
import numpy as np
import os
import mahotas as mh
import cv2
from detectron2.structures import BoxMode
import itertools
import matplotlib as plt
import scipy.misc
from PIL import Image
import shutil
import re
import statistics as st


# Import functions
from Functions import create_dirs, check_sky, mask_seg, count_animals, smooth_animals, plot_image, get_centers_through_borders, get_bboxes, write_file, remove_bad_images


# Set parameters
sigma = 2
minimum_animal_size = 10
kernel = (5,5)
date = "2019-11/"
width = 512
height= 512
output_location = "../Data/labels/" + date #+ "/"


# Create the data directories
#create_dirs()



# Set train locations
input_location = "../Data/images/train/"  #+ "/test/"
input_location_s = "../Data/semantic/train/"  #+ "/test/"
bad_image_list = []

# Run over semantic files and create ground truth train
with open("../Data/labels/train.txt","w") as file: 
    for image_name in os.listdir(input_location_s): # + "/test"

        if image_name == "test":
            continue
            
            
        # Input image from directory
        print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
        input_image_s = input_location_s + image_name
        input_image = input_location + image_name
        
        
        # If image contains sky, i.e. it's under the ground level, continue, else, write the file to the 00_SPLIT file
        if check_sky(input_image_s) == True:
            print("Warning, image contains sky. Removing from set.")
            bad_image_list.append(image_name)
            continue
        
        
        
        # Mask everything but animals in image
        animals, mask = mask_seg(input_image_s)


        
        # Smooth animals with gaussian and remove tiny animals less than set size
        animals_smooth = smooth_animals(animals, sigma = sigma)

        
        # Count animals, if no animals are present, skip image
        labeled_animals, nr_objects, mask_animals = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel)
        #plt.image.imsave('../Data/masks/2019-11/' + image_name, mask_animals)
        #scipy.misc.imsave('../Data/masks/2019-11/' + image_name, mask_animals)
        
        if nr_objects == 0:
            print("Zero animals in this picture, not adding file information to labels file", "\n")
            continue
        else:
            print("animals found")

            # Save masks and animal images
            im = Image.fromarray(mask_animals)
            im.save('../Data/masks/' + image_name)
            output_image = "../Data/images_with_animals/" + image_name 
            shutil.copyfile(input_image, output_image)
        

        # Get centers of animals using boundaries
        centers_list = get_centers_through_borders(labeled_animals, nr_objects, width = width, height = height)
        
        
        # Get bboxes of animals in image
        #centers_list = get_centers(animals_smooth,clean_distance = clean_distance) # old
        bbox_list, bbox_dict_list = get_bboxes(labeled_animals, width = width, height = height)

        
        # Output centers and bboxes
        write_file(output_location,image_name,centers_list,bbox_list)

        annotations = ''
        for dicto in bbox_dict_list:
            annotations += (' ' + str(dicto.get('x0')) + ' ' + str(dicto.get('y0')) + ' ' + str(dicto.get('x1')) + ' ' + str(dicto.get('y1')) + ' ' + "1" + ' ')
        file.write(image_name + annotations)
        file.write("\n")

# Deletes images that have innapropriate compositions in them
remove_bad_images(bad_image_list,input_location,input_location_s)
print("\n Train images done \n")



# Set val locations
input_location = "../Data/images/val/"  #+ "/test/"
input_location_s = "../Data/semantic/val/"  #+ "/test/"
bad_image_list = []

# Run over semantic files and create ground truth test
with open("../Data/labels/val.txt","w") as file: 
    for image_name in os.listdir(input_location_s): # + "/test"

        if image_name == "test":
            continue
            
            
        # Input image from directory
        print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
        input_image_s = input_location_s + image_name
        input_image = input_location + image_name
        
        
        # If image contains sky, i.e. it's under the ground level, continue, else, write the file to the 00_SPLIT file
        if check_sky(input_image_s) == True:
            print("Warning, image contains sky. Removing from set.")
            bad_image_list.append(image_name)
            continue
        
        
        
        # Mask everything but animals in image
        animals, mask = mask_seg(input_image_s)


        
        # Smooth animals with gaussian and remove tiny animals less than set size
        animals_smooth = smooth_animals(animals, sigma = sigma)

        
        # Count animals, if no animals are present, skip image
        labeled_animals, nr_objects, mask_animals = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel)
        #plt.image.imsave('../Data/masks/2019-11/' + image_name, mask_animals)
        #scipy.misc.imsave('../Data/masks/2019-11/' + image_name, mask_animals)
        
        if nr_objects == 0:
            print("Zero animals in this picture, not adding file information to labels file", "\n")
            continue
        else:
            print("animals found")

            # Save masks and animal images
            im = Image.fromarray(mask_animals)
            im.save('../Data/masks/' + image_name)
            output_image = "../Data/images_with_animals/" + image_name 
            shutil.copyfile(input_image, output_image)
        

        # Get centers of animals using boundaries
        centers_list = get_centers_through_borders(labeled_animals, nr_objects, width = width, height = height)
        
        
        # Get bboxes of animals in image
        #centers_list = get_centers(animals_smooth,clean_distance = clean_distance) # old
        bbox_list, bbox_dict_list = get_bboxes(labeled_animals, width = width, height = height)

        
        # Output centers and bboxes
        write_file(output_location,image_name,centers_list,bbox_list)

        annotations = ''
        for dicto in bbox_dict_list:
            annotations += (' ' + str(dicto.get('x0')) + ' ' + str(dicto.get('y0')) + ' ' + str(dicto.get('x1')) + ' ' + str(dicto.get('y1')) + ' ' + "1" + ' ')
        file.write(image_name + annotations)
        file.write("\n")

# Deletes images that have innapropriate compositions in them
remove_bad_images(bad_image_list,input_location,input_location_s)
print("\n Val images done \n")





# Create dicts for dataloader
#get_animal_dicts(input_location , input_location_s , Detectron2_bbox_dict)

# from detectron2.data import DatasetCatalog, MetadataCatalog
# for d in ["2019-11/test/"]:
#     DatasetCatalog.register("../Data/images/" + d, lambda d=d: get_animal_dicts(input_location, input_location_s, Detectron2_bbox_dict))
#     MetadataCatalog.get("../Data/images/" + d).set(thing_classes=["animal"])
#     MetadataCatalog.get("../Data/images/" + d).set(stuff_classes=["background"])
# animal_metadata = MetadataCatalog.get(input_location)

# import random


# dataset_dicts = get_animal_dicts("../Data/images/2019-11/test/", "../Data/semantic/2019-11/test/", Detectron2_bbox_dict)
# print(dataset_dicts)
# for e in random.sample(dataset_dicts, 1):
#     img = cv2.imread(e["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=animal_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(e)
#     cv2.imshow(e,vis.get_image()[:, :, ::-1],)