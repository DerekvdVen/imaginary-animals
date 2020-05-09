from __future__ import print_function

import os
import argparse
import numpy as np

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

#dist = "30m/"
#blur = 1
#bordersfixed
#crop works better
# ALSO TRAINING ON NO ANIMAL IMAGES

# parser for parsing arguments giving in command line file or terminal
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('-n', default="test", type=str, help='run name') # set default to 20304060m
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('-bs', default=8, type=int, help='batchsize')
parser.add_argument('-ne', default=50,type=int,help='number of epochs') # set default to lets say 5 epochs
parser.add_argument('-dir', default="2346m",type=str,help='distance directory of images')
parser.add_argument('-ea', action='store_true', help='save empty animal images to files')
parser.add_argument('-ni', default='', type=str, help='number of images to put in train_x for finetuning') 
parser.add_argument('-perc', default=10, type=int, help='percentage animals in finetuning set') 
parser.add_argument('-renim',type=int,default=5000,help='number of rendered images')
parser.add_argument('-savebest',default=True,type= bool,help='save best loss dict true, save loss at the end of training false')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print("args: ", args)

#parameters: 
batchsize = args.bs
n_epochs = args.ne
checkpoint_name = args.n
name = "23456m_np_mixed_" + str(args.renim) #args.n
testname = "23456m_np_mixed"
dir = args.dir
ni = args.ni

train_loss_list = []
test_loss_list = []
test_loss_rendered_list = []

# set comet info
experiment = Experiment(api_key = "dWZFGTbFA4MerKRqXNpWjLh07", project_name = "general", workspace = "derekvdven",display_summary = False)
experiment.set_name(checkpoint_name)

assert torch.cuda.is_available(), 'Error: CUDA not found!'
print("torchk:", torch.cuda.device_count())

#torch.cuda.set_device(1)

print("current:, ",torch.cuda.current_device())  # output: 0
print("name: ", torch.cuda.get_device_name(1))

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)) # from imagenet set
])



############################### this is training and testing on rendered

# trainset_ren = ListDataset(root='../../Data/images/all/' + dir + '/',
#                         list_file='../../Data/labels/train_' + name + ".txt", train=True, transform=transform, input_size=600)
# trainloader = torch.utils.data.DataLoader(trainset_ren, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=trainset_ren.collate_fn)

testset_rendered = ListDataset(root='../../Data/images/all/' + dir + '/',
                        list_file='../../Data/labels/test_' + testname + '_short.txt', train=False, transform=transform, input_size=600)
testloader_rendered = torch.utils.data.DataLoader(testset_rendered, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=testset_rendered.collate_fn)


################################ this is training and testing on real images

trainset = ListDataset(root='../../Data/kuzikus_patches_800x600/images/',
                        list_file='../../Data/kuzikus_patches_800x600/labels/train' + ni + '.txt', train=True, transform=transform, input_size=600)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=trainset.collate_fn)

testset = ListDataset(root='../../Data/kuzikus_patches_800x600/images/',
                        list_file='../../Data/kuzikus_patches_800x600/labels/test_short_new.txt', train=False, transform=transform, input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)

# concatonate:
# from torch.utils.data import ConcatDataset

# concatDataset = ConcatDataset([trainset_ren, trainset])

# trainloader = torch.utils.data.DataLoader(concatDataset, batch_size=batchsize, shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)


# Model
net = RetinaNet(num_classes=2)

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
    best_loss = 100
    start_epoch = checkpoint['epoch']


net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
print("device")
print(torch.cuda.device_count())
net.cuda()

#print("summary",summary(net,(3,512,512)))



criterion = FocalLoss(num_classes=2, batch_size = batchsize)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0) #momentum=0.9,sgd
#optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=0,momentum=0.9)

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
    print("test_loss ",test_loss)
    print("best_loss ",best_loss)
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
    if args.savebest:
        print('Saving')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + checkpoint_name + "_end.pth")


# Test on rendered
def test_rendered(epoch):
    print('\nTest_rendered')
    net.eval()
    test_loss_rendered = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader_rendered):
        inputs = Variable(inputs.cuda(), requires_grad=False)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss_rendered += loss.data.item()
        avg_loss = test_loss_rendered/(batch_idx + 1)
        print('test_loss_rendered: %.3f | avg_loss: %.3f' % (loss.data.item(), avg_loss))
    test_loss_rendered_list.append(avg_loss)    
    experiment.log_metric("test_loss_rendered", avg_loss)


for epoch in range(start_epoch, start_epoch + n_epochs):
    train(epoch)
    
    test(epoch)

    test_rendered(epoch)

with open("../../output/graph_lists/" + checkpoint_name + "_train_losses.txt","w") as file:
    for item in train_loss_list:
        file.write(str(item))
        file.write("\n")

with open("../../output/graph_lists/" + checkpoint_name + "_test_losses.txt","w") as file:
    for item in test_loss_list:
        file.write(str(item))
        file.write("\n")

with open("../../output/graph_lists/" + checkpoint_name + "_test_rendered_losses.txt","w") as file:
    for item in test_loss_rendered_list:
        file.write(str(item))
        file.write("\n")
