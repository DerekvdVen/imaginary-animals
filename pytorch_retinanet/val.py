from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset_val

from torch.autograd import Variable

from encoder import DataEncoder 

from inference import inference

checkpoint_name = "30m60m_ckpt_7"
dist = "30m60m/"
batchsize = 1
val_loss_list = []

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'

start_epoch = 0

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# "/mnt/guanabana/raid/data/datasets/Kuzikus/bkellenb/annotations/textFiles/"
#"/mnt/guanabana/raid/data/datasets/Kuzikus/bkellenb/annotations/textFiles/val.txt"

#'../../Data/images/all/60m/'
#'../../Data/labels/test.txt'
# CHANGE TO VAL FILES
valset = ListDataset_val(root="/mnt/guanabana/raid/data/datasets/Kuzikus/SAVMAP/data/raster/ebee/",
                        list_file="/mnt/guanabana/raid/data/datasets/Kuzikus/bkellenb/annotations/textFiles/val.txt", transform=transform, input_size=600, bbox_root = "/mnt/guanabana/raid/data/datasets/Kuzikus/bkellenb/annotations/textFiles/")
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=False, num_workers=1)

# Model
net = RetinaNet(num_classes=2)
#net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+ checkpoint_name + ".pth")
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
#net.eval()

#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print("inference")
inference(valset, net, valloader)

