#!/bin/bash

# this code will run the model on a small training set of x real images

run_name="-n 23456z3_2500"                              # name of run
ni="-ni 0"                                                 # number of images
#cp ./checkpoint/20304060m_z.pth ./checkpoint/20304060m_1000f.pth #(only do this for finetuning)
renim="-renim 2500"

dir="-dir 23456m_np"         # 20304060m_np     np means new parameters
perc="-p 10"
lr="--lr 1e-5"                                              # learning rate
bs="-bs 4"                                                  # batch size
ne="-ne 20"                                                 # number of epochs

r="" #!!!!!!!!!!!!                  (fine tuning turn on)

mc="-mc 0.1"                                                # minimum confidence for predicting
iou="-nms_iou 0.2"                                          # non maximal supression iou



python fewer_split_real.py $ni $perc

python train.py $run_name $lr $bs $ne $ni $r $perc $renim
python test.py $run_name $mc $iou