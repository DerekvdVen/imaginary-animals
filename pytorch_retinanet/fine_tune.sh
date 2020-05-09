#!/bin/bash

# this code will run the model on a small training set of x real images

run_name="-n 23456z3_2000e2"                              # name of run
ni="-ni 2000"                                                 # number of images
#cp ./checkpoint/23456z3_looksgood.pth ./checkpoint/23456z3_250fin2.pth #(only do this for finetuning)
renim="-renim 0"                                               #number of rendered images

dir="-dir 23456m_np"         # 20304060m_np     np means new parameters
perc="-p 10"
lr="--lr 1e-5"                                              # learning rate
bs="-bs 4"                                                  # batch size
ne="-ne 20"                                                 # number of epochs

r="" #!!!!!!!!!!!!                  (fine tuning turn on)

mc="-mc 0.01"                                                # minimum confidence for predicting
iou="-nms_iou 0.2"                                          # non maximal supression iou



python fewer_split_real.py $ni $perc

python train.py $run_name $lr $bs $ne $ni $r $perc $renim
python test.py $run_name $mc $iou