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
from utils import change_box_order

#booleans
visualize = False
savePredictions = False
appendConfidences = False
write_to_json = True

#other stuff
imageSize = (4000,3000)
shardsize = (600,600)
stride = 1
batchSize = 1

minConfidence = 0.2
nms_iou = 0.2
colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
#checkpoint = "30m60m_ckpt_4_"
#data_root = 
#save_dir = "../../output/output_images/" + checkpoint

#jsons
gt_dict = {}
pred_dict = {}

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(dataSet, model, dataLoader,output_json,minConfidence=0.2,nms_iou=0.2):
    # setup
    model.eval()
    
    # iterate
    order = np.random.permutation(len(dataSet))
    order = range(len(dataSet)-1)
    
    tBar = trange(len(order))

    count = 0
    for idx in order:
    #for idx, (img, loc_targets, cls_targets, imgPath) in enumerate(dataLoader):
        # count += 1
        # if count == 10:
        #     break
        
        img, loc_targets, cls_targets, imgPath = dataSet.__getitem__(idx)
        #img, imgPath = dataSet.__getitem2__(idx)
        
        print("imgpath: ", imgPath)
        print("targets: ", loc_targets)
        
        #loc_targets = torch.squeeze(loc_targets,dim = 0) # weird output though, so maybe something is wrong with the image order
        tensor = img.to(device) #datagen __getitem__ already transforms

        # evaluate in patches (and batches)
        bboxes = torch.empty(size=(0,4,), dtype=torch.float32)
        labels = torch.empty(size=(0,), dtype=torch.long)
        confs = torch.empty(size=(0, model.num_classes,), dtype=torch.float32) # model.numClasses throws error: RetinaNet object has not attribute 'numClasses'. used 2 instead
        scores = torch.empty(size=(0,), dtype=torch.float32)
        
        gridX, gridY = tensorSharding.createSplitLocations_auto(imageSize, shardsize, tight=True) # changed all this to stuff into what it is in TensorSharding.py
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
        print("all predicted boxes: ", bboxes)
        
        labels = labels[keep]
        #print("labels: ",labels)
        confs = confs[keep,:]
        scores = scores[keep]
        print("scores:  ", scores)
        # update progress bar
        #tBar.set_description_str('# Pred: {}'.format(bboxes.size(0)))
        #tBar.update(1)
        # visualize

        if write_to_json:
            #ground_truth_boxes
            gt_dict[imgPath[-12:]] = loc_targets.tolist()

            #predicted_boxes
            tempdict = {}
            tempdict["boxes"] = bboxes.tolist()
            tempdict["scores"] = scores.tolist()
            pred_dict[imgPath[-12:]] = tempdict


        if visualize:  # and len(bboxes):
            plt.figure(1)
            plt.clf()
            img = dataSet.__getitem2__(idx)
            plt.imshow(img)
            ax = plt.gca()
            for c in range(loc_targets.size(0)):
                ax.add_patch(
                    Rectangle(
                        (loc_targets[c,0], loc_targets[c,1],),
                        loc_targets[c,2]-loc_targets[c,0], loc_targets[c,3]-loc_targets[c,1],
                        fill=False,
                        ec=colors[0]
                    )
                )
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
            #plt.waitforbuttonpress()

            import pylab
            pylab.imshow(img)
            pylab.show()
            
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


    if write_to_json:
        print("writing jsonfile")
        import json
        with open('./calcmeanap/ground_truth_boxes_animals.json', 'w') as fp:
            json.dump(gt_dict, fp)
        with open('./calcmeanap/predicted_boxes_animals.json', 'w') as fp:
            json.dump(pred_dict, fp)    
    if write_to_json:
        print("writing jsonfile")
        import json
        with open('./calcmeanap/' + output_json + '_gt_real.json', 'w') as fp:
            json.dump(gt_dict, fp)
        with open('./calcmeanap/' + output_json + '_pred_real.json', 'w') as fp:
            json.dump(pred_dict, fp)                    