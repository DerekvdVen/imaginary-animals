import rasterio
import rasterio.plot
import pyproj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pylab
import mahotas as mh
import cv2

def create_dirs():
    path_images = "../Data/images/2019-10"
    path_labels = "../Data/labels/2019-10"
    path_semantic = "../Data/semantic/2019-10"
    try:
        os.makedirs(path_images)
        os.makedirs(path_labels)
        os.makedirs(path_semantic)        
    except OSError:
        print ("Creation of the directory %s failed" % path_images)
        print ("Creation of the directory %s failed" % path_labels)
        print ("Creation of the directory %s failed" % path_semantic)
    else:
        print ("Successfully created the directory %s " % path_images)
        print ("Successfully created the directory %s " % path_labels)
        print ("Successfully created the directory %s " % path_semantic)
        
def check_sky(input_image_s):
    img = cv2.imread(input_image_s)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_color_sky1 = np.asarray([59, 177, 180])   # sky
    hsv_color_sky2 = np.asarray([61, 179, 182])   # 

    check_sky = cv2.inRange(img_hsv, hsv_color_sky1, hsv_color_sky2)
    if 255 in check_sky:
        print("Warning, picture contains sky, passing")
        return True
    else:
        return False

def mask_seg(imagelocation):
    
    # Convert to HSV
    img = cv2.imread(imagelocation)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Test if sky is in image

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

    # Perform mask on img_hsv
    target = cv2.bitwise_and(img_hsv,img_hsv, mask=mask)

    # Display image
    #plt.imshow(target)   # this colormap will display in black / white
    #plt.show()
    animals = cv2.bitwise_not(mask)
    
    return animals, mask

def smooth_animals(animals, sigma, size):
    pylab.jet()
    
    sizes = mh.labeled.labeled_size(animals)
    too_small = np.where(sizes < size)
    animals = mh.labeled.remove_regions(animals, too_small)
    
    animalsf = mh.gaussian_filter(animals, sigma)
    animalsf_I = animalsf.astype('uint8')
    T = mh.thresholding.otsu(animalsf_I)
    #pylab.imshow(animalsf)
    #pylab.show()
    
    return animalsf, T

def count_animals(animals_smooth, T,min_size = 1):
    
    # Smooth animals
    labeled, nr_objects = mh.label(animals_smooth > T)
    
    print("This image contains" , nr_objects, "animals")
    
    # plot if animals present
    if nr_objects != 0:
        pylab.imshow(labeled)
        pylab.jet()
        pylab.show()

    return labeled, nr_objects

def get_centers_through_borders(labeled,nr_objects,width = 512, height = 512):
    centers_list = []
    for x in range(nr_objects):
        
        location = np.where(mh.borders(labeled == x))
        x_location = np.mean(location[1])/width
        y_location = np.mean(location[0])/height
        centers_list.append((x_location,y_location))
    print(centers_list)
    return(centers_list)

def get_bboxes(labeled, width = 512, height = 512):
    bboxes = mh.labeled.bbox(labeled)
    bbox_list = []
    
    for box in bboxes[1:]:
        bbox_list.append((abs(box[2] - box[3])/width, abs(box[0] - box[1])/height))
        
    return(bbox_list)

def write_file(output_location,image_name,centers_list,bbox_list):
    bad_image_list1 = []
    output_file = output_location + image_name + ".txt"
    with open(output_file, "w") as file:
        for wh, xy in zip(centers_list, bbox_list):
            if len(centers_list) != len(bbox_list):
                print("Warning: Skipped image, n centers not equal to n animals")
                bad_image_list1.append(image_name)
                break

            file.write("1" +  ' ' + str(wh[0]) + ' ' + str(wh[1]) + ' ' + str(xy[0]) + ' ' + str(xy[1]))
            file.write("\n")
    return(bad_image_list1)

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