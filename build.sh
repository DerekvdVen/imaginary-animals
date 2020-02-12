#!/bin/bash

##### make array of every input: so dir_array = "-dir 30m" "-dir 60m" "-dir 6030m" #####
dir = "-dir 30m"
run_name = "-n test1"
lr = "--lr 1e-5"
bs = "-bs 4"
ne = "-ne 10"
mc = "-mc 0.2"
iou = "-nms_iou 0.2"

# dir_array=(30m 60m 60m30m)

# for x in 0 1 2 
# do
# echo ${dir_array[$x]}
# python check_for_animals ${dir_array[$x]}
# done

# check for animals takes images from Data/all/60m|30m and moves the animal images into Data/only_animal_images/all/60m|30m
echo check all images for animals
python check_for_animals.py $dir

# takes input from Data/all/60m|30m and splits the data into train and test groups in Data/semantic|images/60|30m/ 
echo creating training and validation sets
python split_train_test.py $dir

# takes input from Data/semantic|images/60|30m/ and creates annotations in a train.txt and test.txt file 
echo run main.py with train/test distributions
python main.py 

# runs the CNN with data from Data/all/60|30m and annotations from train.txt and test.txt
echo training retinanet
cd pytorch_retinanet
python train.py $run_name $lr $bs $ne $dir

# check model output with test.py or val.py
python test.py $run_name $mc $iou