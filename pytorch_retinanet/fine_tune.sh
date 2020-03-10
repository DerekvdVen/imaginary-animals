#!/bin/bash

# this code will run the model on a small training set of x real images

run_name="-n 20304060m_1000g"                              # name of run
ni="-ni 1000"                                                 # number of images
cp ./checkpoint/20304060m.pth ./checkpoint/20304060m_1000g.pth



perc="-p 10"
lr="--lr 1e-6"                                              # learning rate
bs="-bs 2"                                                  # batch size
ne="-ne 40"                                                 # number of epochs
r="-r"

mc="-mc 0.1"                                                # minimum confidence for predicting
iou="-nms_iou 0.2"                                          # non maximal supression iou



python fewer_split_real.py $ni $perc

python train.py $run_name $lr $bs $ne $ni $r
python test.py $run_name $mc $iou