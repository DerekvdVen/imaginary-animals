#!/bin/bash

# check for animals takes images from Data/all/60m|30m and moves the animal images into Data/only_animal_images/all/60m|30m
echo generating ground truth
python check_for_animals.py

# takes input from Data/all/60m|30m and splits the data into train and test groups in Data/semantic|images/60|30m/ 
echo creating training and validation sets
python split_train_test.py

# takes input from Data/semantic|images/60|30m/ and creates annotations in a train.txt and test.txt file 
echo rerun main.py with new train/test distributions
python main.py

# runs the CNN with data from Data/all/60|30m and annotations from train.txt and test.txt
echo training retinanet
cd pytorch_retinanet
python train.py

