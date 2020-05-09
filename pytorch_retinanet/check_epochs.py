# check epochs
import os
import torch

with open("./checkpoint/epochs.txt","w") as outfile:
    for x in os.listdir("./checkpoint/"):
        try:
            checkpoint = torch.load('./checkpoint/'+ x)
            start_epoch = checkpoint['epoch']
            print(x)
            print(start_epoch)
            list1 = [x, start_epoch]
            outfile.write(list1)
        except:
            pass