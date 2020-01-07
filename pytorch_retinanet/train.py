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
#experiment = Experiment(api_key = "dWZFGTbFA4MerKRqXNpWjLh07", project_name = "general", workspace = "derekvdven")
#experiment.set_name("60m_6")

dist = "60m/"
#### no pretrained net.py is used this time ####

#parameters: 
batchsize = 2
n_epochs = 200
checkpoint_name = "60m_ckpt6"
train_loss_list = []
test_loss_list = []

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

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
trainset = ListDataset(root='../../Data/images/all/60m/',
                        list_file='../../Data/labels/train.txt', train=True, transform=transform, input_size=600)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=trainset.collate_fn)

testset = ListDataset(root='../../Data/images/all/60m/',
                        list_file='../../Data/labels/test.txt', train=False, transform=transform, input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)


# Model
net = RetinaNet(num_classes=2)
#net.load_state_dict(torch.load('./model/net.pth')) #(net.pth is trained with 20 classes so I cannot load it and use it..)
print(net)

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/'+ checkpoint_name + ".pth")
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

print(summary(net,(3,512,512)))
# load net loadstatedict to change conv2d441 from 180 to 18 and then it should work? 
net.classifier[441] = nn.conv2d[-1,18,4,4]



criterion = FocalLoss(num_classes=2)
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

with open("../../output/graph_lists/" + checkpoint_name + "/train.txt","w") as file:
    for item in train_loss_list:
        file.write(str(item))
        file.write("\n")

with open("../../output/graph_lists/" + checkpoint_name + "/test.txt","w") as file:
    for item in test_loss_list:
        file.write(str(item))
        file.write("\n")
