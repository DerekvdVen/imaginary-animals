from __future__ import print_function

import os
import argparse
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from encoder import DataEncoder
encoder = DataEncoder()
from torch.autograd import Variable

import numpy as np

import TensorSharding as tensorSharding
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utils


#booleans
visualize = True
savePredictions = False
appendConfidences = False

#other stuff
imageSize = (4000,3000)
shardsize = (512,512)
stride = 1
batchSize = 1
minConfidence = 0.05
nms_iou = 0.5
colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
#data_root = 
#save_dir = 

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(dataSet, model):
    # setup
    model.eval()
    
    # iterate
    order = np.random.permutation(len(dataSet))
    tBar = trange(len(order))
    for idx in order:
        img, imgPath = dataSet.__getitem2__(idx)
        print(img, imgPath)
        tensor = transform(img).to(device)

        # evaluate in patches (and batches)
        bboxes = torch.empty(size=(0,4,), dtype=torch.float32)
        labels = torch.empty(size=(0,), dtype=torch.long)
        confs = torch.empty(size=(0, model.num_classes,), dtype=torch.float32) # model.numClasses throws error: RetinaNet object has not attribute 'numClasses'. used 2 instead
        scores = torch.empty(size=(0,), dtype=torch.float32)
        
        gridX, gridY = tensorSharding.createSplitLocations_auto(img.size, shardsize, tight=True) # changed all this to stuff into what it is in TensorSharding.py
        tensors = tensorSharding.splitTensor(tensor, shardsize, gridX, gridY)
        gridX = torch.from_numpy(gridX)
        gridY = torch.from_numpy(gridY)
        gridX, gridY = gridX.view(-1).float(), gridY.view(-1).float()
        
        numPatches = tensors.size(0)
        numBatches = int(np.ceil(numPatches / float(batchSize)))
        
        for t in range(numBatches):
            startIdx = t*batchSize
            endIdx = min((t+1)*batchSize, numPatches)
            
            batch = tensors[startIdx:endIdx,:,:,:].to(device) # added to device here # it's running over the split images
            
            if len(batch.size())==3:
                batch = batch.unsqueeze(0)
            with torch.no_grad():
                bboxes_pred_img, labels_pred_img = model(batch)
                

            bboxes_pred_img, labels_pred_img, confs_pred_img = encoder.decode(loc_preds = bboxes_pred_img.squeeze(0).cpu(),
                                                cls_preds = labels_pred_img.squeeze(0).cpu(),
                                                input_size = (shardsize[1],shardsize[0],),
                                                cls_thresh=minConfidence, nms_thresh=0,
                                                return_conf=True)

            # incorporate patch offsets and append to list of predictions
            if len(bboxes_pred_img):
                bboxes_pred_img[:,0] += gridX[startIdx:endIdx]
                bboxes_pred_img[:,1] += gridY[startIdx:endIdx]
                bboxes_pred_img[:,2] += gridX[startIdx:endIdx]
                bboxes_pred_img[:,3] += gridY[startIdx:endIdx]
                scores_pred_img, _ = torch.max(confs_pred_img,1)
                bboxes = torch.cat((bboxes, bboxes_pred_img), dim=0)
                labels = torch.cat((labels, labels_pred_img), dim=0)
                confs = torch.cat((confs, confs_pred_img), dim=0)
                scores = torch.cat((scores, scores_pred_img), dim=0)
        # do NMS on entire set
        keep = utils.box_nms(bboxes, scores, threshold= nms_iou)
        bboxes = bboxes[keep,:]  
        labels = labels[keep]
        confs = confs[keep,:]
        scores = scores[keep]
        # update progress bar
        tBar.set_description_str('# Pred: {}'.format(bboxes.size(0)))
        tBar.update(1)
        # visualize
        if visualize:  # and len(bboxes):
            plt.figure(1)
            plt.clf()
            plt.imshow(img)
            ax = plt.gca()
            for b in range(bboxes.size(0)):
                ax.add_patch(
                    Rectangle(
                        (bboxes[b,0], bboxes[b,1],),
                        bboxes[b,2]-bboxes[b,0], bboxes[b,3]-bboxes[b,1],
                        fill=False,
                        ec=colors[labels[b]]
                    )
                )
                plt.text(bboxes[b,0], bboxes[b,1], '{:.2f}'.format(scores[b]))
            plt.title('[{}/{}] {}'.format(idx, len(dataSet), imgPath))
            plt.draw()
            plt.waitforbuttonpress()
        # save
        if savePredictions and len(bboxes):
            sz = img.size
            fileName = imgPath.replace(dataRoot, '').replace('.JPG', '.txt').replace('.jpg', '.txt').replace('.NEF', '.txt').replace('.nef', '.txt')
            filePath = os.path.join(saveDir, fileName)
            os.makedirs(os.path.split(filePath)[0], exist_ok=True)
            with open(filePath, 'w') as outFile:
                for b in range(len(bboxes)):
                    # convert to YOLO format
                    w = (bboxes[b,2] - bboxes[b,0]) / sz[0]
                    h = (bboxes[b,3] - bboxes[b,1]) / sz[1]
                    x = (bboxes[b,0] / sz[0]) + w/2.0
                    y = (bboxes[b,1] / sz[1]) + h/2.0
                    label = labels[b]
                    outFile.write('{} {} {} {} {}'.format(label, x, y, w, h))
                    if appendConfidences:
                        outFile.write(' ' + ' '.join([str(c.item()) for c in confs[b,:]]))
                    outFile.write('\n')
    tBar.close()