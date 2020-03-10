#!/bin/bash

# run mulitple models over columns. Output jsons are saved in pytorch_retinanet/calcmeanap/[run_name].json
run_name_array=(t1_30m_ea_false t2_60m t3_6030m)    # name of run
dir_array=(30m 60m 60m30m)                          # data directory

ea_array=(' ' ea ea)                                # save empty animals to train val test list.txts

lr_array=(1e-5 1e-5 1e-5)                           # learning rate
bs_array=(4 4 4)                                    # batch size
ne_array=(50 50 50)                                 # number of epochs

mc_array=(0.2 0.2 0.2)                              # minimum confidence for predicting 
iou_array=(0.2 0.2 0.2)                             # non maximal supression iou


for x in 0 1
do
run_name_x="-n ${run_name_array[$x]}"
dir_x="-dir ${dir_array[$x]}"
ea_x = "-ea ${ea_array[$x]}"
lr_x="--lr ${lr_array[$x]}"
bs_x="-bs ${bs_array[$x]}"
ne_x="-ne ${ne_array[$x]}"
mc_x="-mc ${mc_array[$x]}"
iou_x="-nms_iou ${iou_array[$x]}"

######################################################## Creating annotations
# # check for animals takes images from Data/all/60m|30m and moves the animal images into Data/only_animal_images/all/60m|30m
# echo check all images for animals
# python check_for_animals.py $dir_x

# # takes input from Data/all/60m|30m and splits the data into train and test groups in Data/semantic|images/60|30m/ 
# # make sure to rerun this one when using different distances in dir_x array
echo creating training and validation sets
python split_train_test.py $dir_x

# # takes input from Data/semantic|images/60|30m/ and creates annotations in a train.txt and test.txt file 
echo run main.py with train/test distributions
python main.py $run_name_x $ea_x

######################################################### CNN
# runs the CNN with data from Data/all/60|30m and annotations from train.txt and test.txt
echo training retinanet
cd pytorch_retinanet
python train.py $run_name_x $lr_x $bs_x $ne_x $dir_x $ea_x

# check model output with test.py or val.py
python test.py $run_name_x $mc_x $iou_x

# do the same for val.py 
python val.py $run_name_x $mc_x $iou_x -r
done
