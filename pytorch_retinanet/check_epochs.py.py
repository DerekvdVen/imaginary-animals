# check epochs
import os
import torch

for x in os.listdir("./checkpoint/"):
    checkpoint = torch.load('./checkpoint/'+ x)
    start_epoch = checkpoint['epoch']
    print(x)
    print(start_epoch)