'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from encoder import DataEncoder # .encoder
from transform import resize, random_flip, random_crop, center_crop # .transform
from utils import change_box_order

class ListDataset_val(data.Dataset):
    def __init__(self, root, list_file, transform, input_size, bbox_root):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.transform = transform
        self.input_size = input_size
        self.bbox_root = bbox_root

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines[1:]:
            splitted = line.strip().split()
            self.fnames.append(splitted[2]) # changed 0 to 2 
            #num_boxes = (len(splitted) - 1) // 5
            
            posbox = splitted[3]
            if posbox != "-1":
                with open(os.path.join(bbox_root, posbox)) as f:
                    boxlines = f.readlines()
                    self.num_boxes = len(boxlines)
                box = []
                label = []
                for i in range(self.num_boxes):
                    box_coords = boxlines[i].split(' ')
                    xmid = float(box_coords[0])
                    ymid = float(box_coords[1])
                    width = float(box_coords[2])
                    height = float(box_coords[3])

                    #boxi = torch.tensor([[xmid,ymid,width,height]])
                    #box.append(change_box_order(boxi,'xywh2xyxy'))
                    
                    xyxy_box = [float(xmid - width/2), float(ymid - height/2), float(xmid + width/2), float(ymid + height/2)]
                    box.append(xyxy_box)
                    
                    label.append(1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        img = self.transform(img)
        return img, boxes, labels
    
    def __getitem2__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          img_dir: img location 
        '''
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        img_dir = str(os.path.join(self.root, fname))
        return img, img_dir
    
    def __len__(self):
        return self.num_samples

class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation. TOdo: add blurring (scikit)
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))
        
        # TODo: visualise image and bounding boxes and check, it should be left top right bottom
        # add 30 meter renderings to the training and test dataset
        
        draw = ImageDraw.Draw(img)

        print(boxes)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        img.show()
        import pylab
        pylab.imshow(img)
        pylab.show()
    
        
        img = self.transform(img)
        return img, boxes, labels
    
    def __getitem2__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          img_dir: img location 
        '''
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        img_dir = str(os.path.join(self.root, fname))
        return img, img_dir
    
    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    dataset = ListDataset(root='/mnt/hgfs/D/download/PASCAL_VOC/voc_all_images',
                          list_file='./data/voc12_train.txt', train=True, transform=transform, input_size=600)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

# test()



def test_derek():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    
    dataset = ListDataset(root='../thesis/Data/images/all/',
                          list_file='../thesis/Data/labels/train.txt', train=True, transform=transform, input_size=600)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

