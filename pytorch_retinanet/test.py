
# imports

import os
import argparse

import torch
import torchvision.transforms as transforms

from retinanet import RetinaNet
from encoder import DataEncoder

import utils
from Kuzikus_bigImageValidation import evalOnBigTensor

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from torch.autograd import Variable

# parser for parsing arguments giving in command line file or terminal
parser = argparse.ArgumentParser(description='Testing on rendered images')
parser.add_argument('-n', default="20304060m", type=str, help='checkpoint name')
parser.add_argument('-mc', default=0.1, type=float, help='minConfidence')
parser.add_argument('-nms_iou', default=0.1, type=float, help='nms iou')
args = parser.parse_args()
print("args: ",args)

checkpoint = args.n
write_to_json = False
visualize = True
save_plots = False
colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
minConfidence = args.mc
nms_iou = args.nms_iou

#jsons
gt_dict = {}
pred_dict = {}

if save_plots and not os.path.exists("../../output/output_images/" + args.n + "_" + str(minConfidence)):
    os.mkdir("../../output/output_images/" + args.n + "_" + str(minConfidence))
#dir = "/mnt/guanabana/raid/data/datasets/Kuzikus/SAVMAP/data/raster/ebee/2014-05/20140515_11_rgb/img/"
#outdir = '../../output/output_images/' + checkpoint
#dir = "../../Data/real/"
#dir = '../../Data/only_animal_images/val/'
# "/mnt/guanabana/raid/data/datasets/Kuzikus/SAVMAP/data/raster/ebee/2014-05/20140515_11_rgb/img/"

print('Loading model..')
net = RetinaNet(num_classes=2)
net.load_state_dict(torch.load('./checkpoint/' + checkpoint + ".pth")['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])






# these are for rendered results

# dir = '../../Data/images/val/' # this will be where the val images are
# val_loc = "../../Data/labels/val_" + checkpoint + '.txt'
# size = w = h = 600
# origsize = origw = origh = 512
# changeboxtypebool = False
############################################################

# these are for doing a quick check of the rendered train model on anther val set without splicing and stuff
#val_animals.txt
dir = "../../Data/kuzikus_patches_800x600/images/"
val_loc = "../../Data/kuzikus_patches_800x600/labels/val_animals.txt" # test on smaller things
w = 600
h = 600
origw = 800
origh = 600
changeboxtypebool = True
############################################################

count = 0
with open(val_loc) as annotations_file: # this will be where the val.txt file is. 
    for line in annotations_file:
        # count = count+1
        # if count > 300:
        #     break

        line_list = line.strip().split(' ')
        img_x = line_list[0]
        
        boxesdata = line_list[1:]
        nboxes = int(len(boxesdata)/5)
        loc_targets = []
        cls_targets = []

        # recast loc_targets to the new size image
        for i in range(0,len(boxesdata),5):

            #for square image
            #loc_targets.append([round(float(j)/origsize*size,4) for j in boxesdata[i:i+4]])
            centerx = float(boxesdata[i])
            #print("centerx: ",centerx)
            centery = float(boxesdata[i+1])
            boxwidth = float(boxesdata[i+2])
            #print("boxwidth", boxwidth)
            boxheight = float(boxesdata[i+3])
            
            if changeboxtypebool == True:
                box1 = centerx - boxwidth/2
                box2 = centery - boxheight/2
                box3 = centerx + boxwidth/2
                box4 = centery + boxheight/2
                if box1 < 0:
                    box1 = 0
                if box2 < 0:
                    box2 = 0
                if box3 > origw:
                    box3 = origw
                if box4 > origh:
                    box4 = origh
            else:
                box1 = float(boxesdata[i])
                box2 = float(boxesdata[i+1])
                box3 = float(boxesdata[i+2])
                box4 = float(boxesdata[i+3])
            
            box = []
            box.append(round(float(box1)/origw*w,4))
            box.append(round(float(box2)/origh*h,4))
            box.append(round(float(box3)/origw*w,4))
            box.append(round(float(box4)/origh*h,4))
            #print("boxes after change: ", box)
            loc_targets.append(box)
            cls_targets.append(int(boxesdata[i+4]))

        #print("loctargets: ", loc_targets)
        #print("clstargets: ", cls_targets)
        print(img_x)

        image = Image.open(dir + img_x).convert('RGB')
        image = image.resize((w,h))
        
        #print('Predicting..')
        x = transform(image)
        x = x.unsqueeze(0)
        x = Variable(x, requires_grad = False)
        loc_preds, cls_preds = net(x) 
        
        # decode loc preds and cls preds
        print('Decoding..')
        encoder = DataEncoder()
        boxes_pred_img, labels_pred_img, confs_pred_img = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), input_size =(w,h,) ,cls_thresh = minConfidence, nms_thresh=0.5, return_conf=True)

        # set empty tensors
        bboxes = torch.empty(size=(0,4,), dtype=torch.float32)  
        labels = torch.empty(size=(0,), dtype=torch.long)
        confs = torch.empty(size=(0, net.num_classes,), dtype=torch.float32) 
        scores = torch.empty(size=(0,), dtype=torch.float32)
        
        # add output to empty tensors
        if len(boxes_pred_img):
            scores_pred_img, _ = torch.max(confs_pred_img,1)
            bboxes = torch.cat((bboxes, boxes_pred_img), dim=0)
            labels = torch.cat((labels, labels_pred_img), dim=0)
            confs = torch.cat((confs, confs_pred_img), dim=0)
            scores = torch.cat((scores, scores_pred_img), dim=0)
        
        # perform nms
        if len(bboxes):
            print("before nms iou boxes: ", bboxes)
            print("before nms iou confs: ", confs)
        keep = utils.box_nms(bboxes, scores, threshold= nms_iou)
        bboxes = bboxes[keep,:]  
        labels = labels[keep]
        confs = confs[keep,:]
        scores = scores[keep]

        # calculate statistics from these predicted boxes and labels, and the ground truth boxes and labels
        if len(loc_targets):
            print("predicted boxes: ", bboxes)
            print("predicted scores: ",scores)
            print("labels: ",labels)

        if write_to_json:

            #ground_truth_boxes
            gt_dict[img_x] = loc_targets #tolist 

            #predicted_boxes
            tempdict = {}
            tempdict["boxes"] = bboxes.tolist()
            tempdict["scores"] = scores.tolist()
            pred_dict[img_x] = tempdict

        if visualize and len(loc_targets):
            plt.figure(1)
            plt.clf()
            plt.imshow(image)
            ax = plt.gca()
            for c in range(len(loc_targets)):
                ax.add_patch(
                    Rectangle(
                        (loc_targets[c][0], loc_targets[c][1],),
                        loc_targets[c][2]-loc_targets[c][0], loc_targets[c][3]-loc_targets[c][1],
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
            #plt.title('[{}/{}] {}'.format(idx, len(dataSet), imgPath))
            plt.draw()
            #plt.waitforbuttonpress()

            import pylab
            pylab.imshow(image)
            pylab.show()
        
        if save_plots and len(loc_targets):
            plt.figure(1)
            plt.clf()
            plt.imshow(image)
            plt.axis('off')
            ax = plt.gca()
            for c in range(len(loc_targets)):
                ax.add_patch(
                    Rectangle(
                        (loc_targets[c][0], loc_targets[c][1],),
                        loc_targets[c][2]-loc_targets[c][0], loc_targets[c][3]-loc_targets[c][1],
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
            #plt.title('[{}/{}] {}'.format(idx, len(dataSet), imgPath))
            plt.draw()
            plt.savefig("../../output/output_images/"+ args.n + "_" + str(minConfidence) + "/" + img_x.replace("/","-"),bbox_inches = 'tight',pad_inches=0)
        #print("plot the thing")

if write_to_json:
        print("writing jsonfile")
        import json
        with open('./calcmeanap/' + args.n + '_full_' + str(args.mc) + '_gt.json', 'w') as fp:
            json.dump(gt_dict, fp)
        with open('./calcmeanap/' + args.n + '_full_' + str(args.mc) +'_pred.json', 'w') as fp:
            json.dump(pred_dict, fp)            