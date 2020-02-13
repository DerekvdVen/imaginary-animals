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
from PIL import Image, ImageDraw
import pylab
import argparse

# Import functions
from Functions import create_dirs, check_sky, mask_seg, count_animals, smooth_animals, plot_image, get_centers_through_borders, get_bboxes, write_file, remove_bad_images

# 60m: minimum animal size = 20?

# Set parameters
sigma = 2
minimum_animal_size = 25
kernel = (7,7)
width = height = 512

parser = argparse.ArgumentParser(description='Creating annotations')
parser.add_argument('-ea', action='store_true', help='save empty animal images to files')
args = parser.parse_args()

save_empty_images_bool = args.ea
print("args: ",args)
import time
time.sleep(2)

plotbool = False
rembordbool = False

# Create the data directories
#create_dirs()

bad_image_list = []

# Set train locations
input_location_train = "../Data/images/train/"  #+ "/test/"
input_location_s_train = "../Data/semantic/train/"  #+ "/test/"

# Set test locations
input_location_test = "../Data/images/test/"  #+ "/test/"
input_location_s_test = "../Data/semantic/test/"  #+ "/test/"

# Set vallocations
input_location_val = "../Data/images/val/"  #+ "/test/"
input_location_s_val = "../Data/semantic/val/"  #+ "/test/"

def main(label_loc,input_location,input_location_s):
    with open(label_loc,"w") as file: 
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
            labeled_animals, nr_objects = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel,plot = plotbool,removeborder=rembordbool)
            
            if nr_objects == 0:
                print("Zero animals in this picture, not adding file information to labels file", "\n")
                if save_empty_images_bool == True:
                    file.write(image_name)
                    file.write("\n")
                continue

            # Get centers of animals using boundaries
            centers_list = get_centers_through_borders(labeled_animals, nr_objects, width = width, height = height)
            
            # Get bboxes of animals in image
            #centers_list = get_centers(animals_smooth,clean_distance = clean_distance) # old
            bbox_list, bbox_dict_list = get_bboxes(labeled_animals, width = width, height = height)

            # Output centers and bboxes
            #write_file(output_location,image_name,centers_list,bbox_list)

            annotations = ''
            for dicto in bbox_dict_list:
                annotations += (' ' + str(dicto.get('x0')) + ' ' + str(dicto.get('y0')) + ' ' + str(dicto.get('x1')) + ' ' + str(dicto.get('y1')) + ' ' + "1")
            file.write(image_name + annotations)
            file.write("\n")
            
            # img = Image.open(input_image)
            # if img.mode != 'RGB':
            #     img = img.convert('RGB')
            # draw = ImageDraw.Draw(img)

            # print(annotations)
            # #for box in annotations:
            # #    draw.rectangle(list(box), outline='red')
            # img.show()
            # import pylab
            # pylab.imshow(img)
            # pylab.show()

            print("img", image_name)
            print("img_s", input_image_s)
            #plot_image(input_image_s)
    
    # Deletes images that have innapropriate compositions in them
    remove_bad_images(bad_image_list,input_location,input_location_s)

main("../Data/labels/train_all.txt", input_location_train, input_location_s_train)
main("../Data/labels/test_all.txt", input_location_test, input_location_s_test)
main("../Data/labels/val_all.txt", input_location_val, input_location_s_val)




# # Run over semantic files and create ground truth test
# with open("../Data/labels/test.txt","w") as file: 
#     for image_name in os.listdir(input_location_s): # + "/test"

#         if image_name == "test":
#             continue
            
            
#         # Input image from directory
#         print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
#         input_image_s = input_location_s + image_name
#         input_image = input_location + image_name
        
        
#         # If image contains sky, i.e. it's under the ground level, continue, else, write the file to the 00_SPLIT file
#         if check_sky(input_image_s) == True:
#             print("Warning, image contains sky. Removing from set.")
#             bad_image_list.append(image_name)
#             continue
        
        
        
#         # Mask everything but animals in image
#         animals, mask = mask_seg(input_image_s)


        
#         # Smooth animals with gaussian and remove tiny animals less than set size
#         animals_smooth = smooth_animals(animals, sigma = sigma)

        
#         # Count animals, if no animals are present, skip image
#         labeled_animals, nr_objects = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel, plot = plotbool)
        
#         if nr_objects == 0:
#             print("Zero animals in this picture, not adding file information to labels file", "\n")
#             continue
        

#         # Get centers of animals using boundaries
#         centers_list = get_centers_through_borders(labeled_animals, nr_objects, width = width, height = height)
        
        
#         # Get bboxes of animals in image
#         #centers_list = get_centers(animals_smooth,clean_distance = clean_distance) # old
#         bbox_list, bbox_dict_list = get_bboxes(labeled_animals, width = width, height = height)

        
#         # Output centers and bboxes
#         write_file(output_location,image_name,centers_list,bbox_list)

#         annotations = ''
#         for dicto in bbox_dict_list:
#             annotations += (' ' + str(dicto.get('x0')) + ' ' + str(dicto.get('y0')) + ' ' + str(dicto.get('x1')) + ' ' + str(dicto.get('y1')) + ' ' + "1")
#         file.write(image_name + annotations)
#         file.write("\n")

# print("\n test images done \n")



# # Run over semantic files and create ground truth val
# with open("../Data/labels/val.txt","w") as file: 
#     for image_name in os.listdir(input_location_s): # + "/test"

#         if image_name == "test":
#             continue
            
            
#         # Input image from directory
#         print("###----------###" + "\n" + image_name + "\n" + "###----------###" + "\n")
#         input_image_s = input_location_s + image_name
#         input_image = input_location + image_name
        
        
#         # If image contains sky, i.e. it's under the ground level, continue, else, write the file to the 00_SPLIT file
#         if check_sky(input_image_s) == True:
#             print("Warning, image contains sky. Removing from set.")
#             bad_image_list.append(image_name)
#             continue
        
        
        
#         # Mask everything but animals in image
#         animals, mask = mask_seg(input_image_s)


        
#         # Smooth animals with gaussian and remove tiny animals less than set size
#         animals_smooth = smooth_animals(animals, sigma = sigma)

        
#         # Count animals, if no animals are present, skip image
#         labeled_animals, nr_objects = count_animals(animals_smooth, minimal_size = minimum_animal_size,image_kernel=kernel, plot = plotbool)
        
#         if nr_objects == 0:
#             print("Zero animals in this picture, not adding file information to labels file", "\n")
#             continue
        

#         # Get centers of animals using boundaries
#         centers_list = get_centers_through_borders(labeled_animals, nr_objects, width = width, height = height)
        
        
#         # Get bboxes of animals in image
#         #centers_list = get_centers(animals_smooth,clean_distance = clean_distance) # old
#         bbox_list, bbox_dict_list = get_bboxes(labeled_animals, width = width, height = height)

        
#         # Output centers and bboxes
#         write_file(output_location,image_name,centers_list,bbox_list)

#         annotations = ''
#         for dicto in bbox_dict_list:
#             annotations += (' ' + str(dicto.get('x0')) + ' ' + str(dicto.get('y0')) + ' ' + str(dicto.get('x1')) + ' ' + str(dicto.get('y1')) + ' ' + "1")
#         file.write(image_name + annotations)
#         file.write("\n")




