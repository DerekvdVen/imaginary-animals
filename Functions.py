# Functions for Main.py from imaginary-animals

import numpy as np
import os
import pylab
import mahotas as mh
import cv2
import copy



def create_dirs():
    
    
    # create_dirs creates directories for storing data
    # input: none
    # output: none
    
    
    path_images = "../Data/images/" 
    path_labels = "../Data/labels/" 
    #path_semantic = "../Data/semantic/" 
    
    try:
        os.makedirs(path_images)
    except OSError:
        print ("Creation of the directory %s failed" % path_images)
    else:
        print ("Successfully created the directory %s " % path_images)

    try:
        os.makedirs(path_labels)   
    except OSError:
        print ("Creation of the directory %s failed" % path_labels)
    else:
        print ("Successfully created the directory %s " % path_labels)
        
    try:
        os.makedirs(path_images)     
    except OSError:
        print ("Creation of the directory %s failed" % path_images)
    else:
        print ("Successfully created the directory %s " % path_images)

                
def check_sky(input_image_s):
    
    
    # check_sky checks if the image contains sky pixels, i.e. if the camera is taking pictures below ground, and deletes them if present
    # input: an input image 
    # output: True (sky) or False (no sky)
    
    # Read image and set sky mask
    img = cv2.imread(input_image_s)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_color_sky1 = np.asarray([59, 177, 180])   # sky
    hsv_color_sky2 = np.asarray([61, 179, 182])   # 
    check_sky = cv2.inRange(img_hsv, hsv_color_sky1, hsv_color_sky2)
    
    # Return true or false based on sky presence
    if 255 in check_sky:
        print("Warning, picture contains sky, passing")
        return True
    else:
        return False
    
    
def mask_seg(imagelocation):
    
    
    # mask_seg masks ground, vegetation and water pixels so we only capture the animals
    # input: location of image to be masked
    # output: animal blobs array, and mask array
    
    
    # Convert to HSV
    img = cv2.imread(imagelocation)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Set exclusion parameters for ground, vegetation and water
    hsv_color_ground1 = np.asarray([50, 210, 224])   # exclude ground
    hsv_color_ground2 = np.asarray([52, 212, 226])   # 

    hsv_color_veg1 = np.asarray([32, 172, 249])   # exclude veg
    hsv_color_veg2 = np.asarray([34, 174, 251])   # 

    hsv_color_water1 = np.asarray([9, 237, 245])   # exclude water
    hsv_color_water2 = np.asarray([11, 239, 247])   # 

    # Create masks based on exclusion parameters and combine them
    mask_ground = cv2.inRange(img_hsv, hsv_color_ground1, hsv_color_ground2)
    mask_veg = cv2.inRange(img_hsv, hsv_color_veg1, hsv_color_veg2)
    mask_water = cv2.inRange(img_hsv, hsv_color_water1, hsv_color_water2)
    mask = mask_ground + mask_veg + mask_water

    # Perform mask on img_hsv, and get animals array
    #target = cv2.bitwise_and(img_hsv,img_hsv, mask=mask)
    animals = cv2.bitwise_not(mask)
    
    return animals, mask


def smooth_animals(animals, sigma):
    
    
    # smooth_animals smoothes the animal blobs
    # input: animal blob array, sigma (which is the standard deviation for Gaussian kernel)
    # output: animalsf: smoothed animal blobs array
    

    # Apply gaussian filter
    animals_smooth = mh.gaussian_filter(animals, sigma)
    
    
    return animals_smooth

def check_for_animals(animals):
    labeled_test, nr_objects_test = mh.label(animals)
    return(nr_objects_test)

def count_animals(animals_smooth,minimal_size,image_kernel,plot):
       
        
    # count_animals thresholds the smoothed gaussian blobs and then labels and counts them. Also makes plots to show the animals.
    # input: animals_smooth: smoothed animal blobs array, plus_factor: modifies otsu threshold to make the blobs a bit smaller
    # output: array with labels, and a number of objects integer
    
    
    # Find Otsu threshold T. This is neccessary because amimals_smooth creates a gaussian distribution which is square when labeled. We don't want these squares to overlap when counting, so we threshold them using the Otsu threshold, which minimizes the intra-class variance.
    animals_smooth_I = animals_smooth.astype('uint8')
    T = mh.thresholding.otsu(animals_smooth_I)
    
    # Make a plot to compare effects of erosion and dilation
    labeled_test, nr_objects_test = mh.label(animals_smooth_I > T)
    
    # Erode
    kernel = np.ones(image_kernel, np.uint8)
    animals_erosion = cv2.erode(animals_smooth_I, kernel, iterations = 1)
    
    # Label animals and print the amount
    labeled, nr_objects = mh.label(animals_erosion > T)
    print("This image contains" , nr_objects, "animals, including tiny blobs")


    if nr_objects != 0:
        
        if plot == True:
            pylab.imshow(labeled)
            pylab.jet()
            pylab.show()

        # remove small sizes
        sizes = mh.labeled.labeled_size(labeled)
        too_small = np.where(sizes < minimal_size)
        labeled = mh.labeled.remove_regions(labeled, too_small)
        print("without small blobs")
        if plot == True:
            pylab.imshow(labeled)
            pylab.jet()
            pylab.show()

        # remove blobs touching the border
        print("without border blobs")
        labeled = mh.labeled.remove_bordering(labeled)
        if plot == True:
            pylab.imshow(labeled)
            pylab.jet()
            pylab.show()

        # Relabel after removing small blobs
        labeled, nr_objects = mh.label(labeled)
        labeled = labeled.astype('uint8')
        
        # Dilate
        labeled = cv2.dilate(labeled, kernel, iterations = 1)
        print("This image contains" , nr_objects, "animals, excluding tiny blobs")
    
    if nr_objects != 0 and plot == True:
        print("Labeled blobs before erosion")
        pylab.imshow(labeled_test)
        pylab.jet()
        pylab.show()
        
        print("Labeled blobs after dilation")
        pylab.imshow(labeled)
        pylab.jet()
        pylab.show()

            
            
    return labeled, nr_objects


def plot_image(image):
    
    
    # plot_image plots an input image
    # input: image in file location
    # output: none
    
    
    # Convert from BGR to RGB and plot
    print("Real image")
    img = cv2.imread(image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pylab.imshow(RGB_img)
    pylab.show()
    
  
    
def get_centers_through_borders(labeled, nr_objects, width = 512, height = 512):
    
    
    # get_centers_through_borders creates a list and writes the centers of the labeled blobs to it
    # input: labeled: labeled animals array, nr_objects in the image, width and height of the image
    # output: list of centers in image
    
    
    # Create centers_list list for centers
    centers_list = []
    
    print("unique things:  ", np.unique(labeled))
    # For amount of animals in picture, find the borders of every animal, and take the mean value of the x and y border values, then append to list
    for x in range(nr_objects):
        
        location = np.where(mh.borders(labeled == x+1))
        print("center location: ",  location)
        x_location = np.mean(location[1])/width
        y_location = np.mean(location[0])/height
        centers_list.append((x_location,y_location))
    
    # Print the center locations list
    print("center locations:" ,  str(centers_list) , "\n")
    
    return(centers_list)


def get_bboxes(labeled, width = 512, height = 512):
    
    
    # get_bboxes creates a list and writes the bouning boxes of the animals of the image to it
    # input: labeled: labeled animals array, width and height of the image
    # output: bbox_list: bounding box list
    
    
    # Calculate bboxes with bbox function and make a list to store the coordinates in
    bboxes = mh.labeled.bbox(labeled)
    bbox_list = []
    bbox_dict_list = []
    print("boxes:  ", bboxes)
    
    # Store coordinates in the bbox list
    for box in bboxes[1:]:
        
        # For Kellenberger
        bbox_list.append((abs(box[2] - box[3])/width, abs(box[0] - box[1])/height))
        
        # For Detectron2
        bbox_dict = {}
        bbox_dict["x0"], bbox_dict["y0"], bbox_dict["x1"], bbox_dict["y1"] = box[2], box[0], box[3], box[1]
        bbox_dict_list.append(bbox_dict)
                                                                                 
    # Print the bbox list
    print("bboxes:", str(bbox_list), "\n")
    
    return(bbox_list, bbox_dict_list)


def write_file(output_location,image_name,centers_list,bbox_list):
    
    
    # write_file writes the image annotation data to a directory
    # input: output_location: place to save, image_name: the image name, centers list and bbox list created earlier
    # output: 
    
    
    # Create output_file name
    output_file = output_location + image_name + ".txt"
    
    # Open the output_file and write data 
    with open(output_file, "w") as file:
        
        for wh, xy in zip(centers_list, bbox_list):
            file.write("1" +  ' ' + str(wh[0]) + ' ' + str(wh[1]) + ' ' + str(xy[0]) + ' ' + str(xy[1]))
            file.write("\n")
        print("Proper animals found: writing file", "\n")    


def remove_bad_images(bad_image_list,input_location,input_location_s):
    print(bad_image_list)
    for image in bad_image_list:
        try:
            os.remove(input_location + image)
            os.remove(input_location_s + image)
            print("removed file")
        except:
            print("File already removed")
        



'''
def get_animal_dicts(img_dir, seg_dir, bboxes):
    
    dataset_dicts_list = []
    img_list = os.listdir(img_dir)
    #seg_list = os.listdir(seg_dir)
    
    for image in img_list:
        
        print(image)
        image_dict = {}
#     file_name: the full path to the image file.
        image_dict["file_name"] = img_dir + image
    
#     sem_seg_file_name: the full path to the ground truth semantic segmentation file.
        image_dict["sem_seg_file_name"] = seg_dir + image

#     image: the image as a numpy array.
        image_dict["image"] = cv2.imread(img_dir + image)
    
#     sem_seg: semantic segmentation ground truth in a 2D numpy array. Values in the array represent category labels.
        image_dict["image"] = cv2.imread(seg_dir + image)
    
#     height, width: integer. The shape of image.
        image_dict["height"], image_dict["width"] = cv2.imread(img_dir + image).shape[:2]
    
#     image_id (str): a string to identify this image. Mainly used by certain datasets during evaluation to identify the image, but a dataset may use it for different purposes.
        image_dict["image_id"] = image[:-4]

#     annotations (list[dict]): the per-instance annotations of every instance in this image. Each annotation dict may contain:
        annotations = []

        if bboxes.get(image) != None:

            for box_dict in bboxes.get(image):
                annotations_dict = {}
                
                #     bbox (list[float]): list of 4 numbers representing the bounding box of the instance.
                annotations_dict["bbox"] = [box_dict.get("x0"), box_dict.get("y0"), box_dict.get("x1"), box_dict.get("y1")] 
                
                #     bbox_mode (int): the format of bbox. It must be a member of structures.BoxMode. Currently supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.
                annotations_dict["bbox_mode"] = "BoxMode.XYXY_ABS"

                #     category_id (int): an integer in the range [0, num_categories) representing the category label. The value num_categories is reserved to represent the “background” category, if applicable.
                annotations_dict["category_id"] = 1
                
                annotations.append(annotations_dict)
       
        
    
    
        image_dict["annotations"] = annotations
        dataset_dicts_list.append(image_dict)
        
        #print(image_dict, "\n")
    return dataset_dicts_list

        
        
        
        
        
        
'''
        
        
        
        
# old stuff
'''
def get_centers(animals_smooth, clean_distance, width = 512, height = 512):
    rmax = mh.regmax(animals_smooth)
    centers, nr_centers = mh.label(rmax)
    print("nr centers", nr_centers)
    
    # plot if animals present
    if nr_centers != 0:
        pylab.imshow(animals_smooth)
        pylab.jet()
        pylab.show()

    centers_list = []
    centers_list_clean = []
    for x in range(nr_centers):
        location = np.where(centers == (x+1))
        print(location)
        
        # if center location is more than one pixel,
        if len(location[0])>1:
            x_location = float(np.mean(location[1]))/width ### check if mean works here, it should
            print("location is big")
        else:
            x_location = float(location[1])/width
            
        if len(location[1])>1:
            y_location = float(np.mean(location[0][0]))/height
            print("big center, refining")
        else:
            y_location = float(location[0])/height
            
        centers_list.append((x_location,y_location))
    
    # This checks if centers are not too close together
    for x in range(0,len(centers_list)-1):        
        if abs(centers_list[x][0] - centers_list[x+1][0]) < clean_distance/width and abs(centers_list[x][1]-centers_list[x+1][1]) < clean_distance/height:
            print("Multiple centers found, removing centers close together")
            print(centers_list[x][0]*width, centers_list[x+1][0]*width,"and",centers_list[x][1]*height,centers_list[x+1][1]*height)
        else:
            centers_list_clean.append(centers_list[x])

    centers_list_clean.append(centers_list[-1])
    
    print("nr centers", len(centers_list_clean))
    print(centers_list_clean)
    return centers_list_clean
'''