#!/bin/bash

run_name="-n test60"       # name of run
dir="-dir 60m"            # data directory

ea="-ea"                  # save empty animals to train val test list.txts

lr="--lr 1e-5"            # learning rate
bs="-bs 1"                # batch size
ne="-ne 1"                # number of epochs

mc="-mc 0.2"              # minimum confidence for predicting
iou="-nms_iou 0.2"        # non maximal supression iou

# # check for animals takes images from Data/all/60m|30m and moves the animal images into Data/only_animal_images/all/60m|30m
# echo check all images for animals
# python check_for_animals.py $dir

# # takes input from Data/all/60m|30m and splits the data into train and test groups in Data/semantic|images/60|30m/ 
# echo creating training and validation sets
# python split_train_test.py $dir

# # takes input from Data/semantic|images/60|30m/ and creates annotations in a train.txt and test.txt file 
# echo run main.py with train/test distributions
# python main.py $run_name $ea

# # runs the CNN with data from Data/all/60|30m and annotations from train.txt and test.txt
# echo training retinanet
cd pytorch_retinanet
python train.py $run_name $lr $bs $ne $dir

# # check model output with test.py or val.py
# echo testing on rendered images
# python test.py $run_name $mc $iou

# # python val.py 
# echo testing on real images
# python val.py $run_name $mc $iou -r