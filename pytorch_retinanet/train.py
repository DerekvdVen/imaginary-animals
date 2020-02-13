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

from torch.autograd import Variable
from torchsummary import summary

import numpy as np

dist = "30m/"
#blur = 1
#bordersfixed
#crop works better
# ALSO TRAINING ON NO ANIMAL IMAGES

# command line arguments
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('-n', default="test1", type=str, help='run name')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('-bs', default=2, type=int, help='batchsize')
parser.add_argument('-ne', default=10,type=int,help='number of epochs')
parser.add_argument('-dir', default="30m",type=str,help='distance directory of images')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print("args: ", args)

#parameters: 
batchsize = args.bs
n_epochs = args.ne
checkpoint_name = args.n
dist = args.dir
train_loss_list = []
test_loss_list = []



# set comet info
experiment = Experiment(api_key = "dWZFGTbFA4MerKRqXNpWjLh07", project_name = "general", workspace = "derekvdven",display_summary = False)
experiment.set_name(checkpoint_name)

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)) # from imagenet set
])

# derek
trainset = ListDataset(root='../../Data/images/all/' + dist + '/',
                        list_file='../../Data/labels/train_all.txt', train=True, transform=transform, input_size=600)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=trainset.collate_fn)

testset = ListDataset(root='../../Data/images/all/' + dist + '/',
                        list_file='../../Data/labels/test_all.txt', train=False, transform=transform, input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)


# Model
net = RetinaNet(num_classes=2)
#net.load_state_dict(torch.load('./model/net.pth')) #(net.pth is trained with 20 classes so I cannot load it and use it..)
#print(net)

ordered_state_dict = torch.load("./model/net.pth")

#print(ordered_state_dict["cls_head.8.weight"].size())
#ordered_state_dict["cls_head.8.weight"] = ordered_state_dict["cls_head.8.weight"][0:18]
#print(ordered_state_dict["cls_head.8.weight"].size())
ordered_state_dict["cls_head.8.weight"] = torch.from_numpy(np.random.randn(18,256,3,3)*np.sqrt(1/256)) #hexavier
#print(ordered_state_dict["cls_head.8.weight"].size())

#print(ordered_state_dict["cls_head.8.bias"].size())
ordered_state_dict["cls_head.8.bias"] = ordered_state_dict["cls_head.8.bias"][0:18]
#ordered_state_dict["cls_head.8.bias"] = torch.ones(18)
#print(ordered_state_dict["cls_head.8.bias"])

#print(ordered_state_dict["cls_head.8.weight"].size())
net.load_state_dict(ordered_state_dict)


if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+ checkpoint_name + ".pth")
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

#print("summary",summary(net,(3,512,512)))



criterion = FocalLoss(num_classes=2, batch_size = batchsize)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0) #momentum=0.9,sgd

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        avg_loss = train_loss/(batch_idx + 1) # batch_idx + 1 == len(trainloader)
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), avg_loss))
    train_loss_list.append(avg_loss)
    experiment.log_metric("train_loss", avg_loss)

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), requires_grad=False)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data.item()
        avg_loss = test_loss/(batch_idx + 1)
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), avg_loss))
    test_loss_list.append(avg_loss)    
    experiment.log_metric("test_loss", avg_loss)

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + checkpoint_name + ".pth")
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch + n_epochs):
    train(epoch)
    
    test(epoch)

# with open("../../output/graph_lists/" + checkpoint_name + "/train.txt","w") as file:
#     for item in train_loss_list:
#         file.write(str(item))
#         file.write("\n")

# with open("../../output/graph_lists/" + checkpoint_name + "/test.txt","w") as file:
#     for item in test_loss_list:
#         file.write(str(item))
#         file.write("\n")
