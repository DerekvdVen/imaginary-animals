#!/bin/bash

echo generating ground truth
python main.py

echo creating training and validation sets
python split_train_test.py

echo training retinanet
cd pytorch_retinanet
python train.py

