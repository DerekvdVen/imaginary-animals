'''Perform transforms on both PIL image and object boxes.'''
import math
import random

import torch
import torchvision.transforms as transforms
import cv2
import skimage.filters
import PIL


from PIL import Image, ImageDraw, ImageFilter


def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    img = img.resize((ow,oh), Image.BILINEAR)
    if len(boxes):
        boxes *= torch.Tensor([sw,sh,sw,sh])
    return img, boxes

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    from copy import deepcopy
    img_orig = deepcopy(img)
    boxes_orig = deepcopy(boxes)

    for _ in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    if len(boxes):
        boxes -= torch.Tensor([x,y,x,y])
        boxes[:,0::2].clamp_(min=0, max=w-1)
        boxes[:,1::2].clamp_(min=0, max=h-1)
    
    # if len(boxes.size()) > 1:
    #     draw2 = ImageDraw.Draw(img)

    #     for box in boxes:
    #         draw2.rectangle(list(box), outline='red')
    #     img.show()
    #     import pylab
    #     pylab.imshow(img)
    #     pylab.show()

    # WRITE CODE TO NOT CROP IF THE BOXES GO OUT OF THE IMAGE
    #print(boxes)
    #print(boxes_orig)
    for box, obox in zip(boxes, boxes_orig):
        #print(box)
        #print(box.data[0].item())
        #print(box.data[1].item())
        
        length = abs(box.data[0].item() - box.data[2].item())
        height = abs(box.data[1].item() - box.data[3].item())
        olength = abs(obox.data[0].item() - obox.data[2].item())
        oheight = abs(obox.data[1].item() - obox.data[3].item())
        # print(length)
        # print(height)
        # print(olength)
        # print(oheight)
        if length < 5 or height < 5:
            print("box is too slim")
            
            # draw2 = ImageDraw.Draw(img_orig)

            # for box1 in boxes_orig:
            #     draw2.rectangle(list(box1), outline='red')
            # img_orig.show()
            # import pylab
            # pylab.imshow(img_orig)
            # pylab.show()
            return img_orig, boxes_orig
        elif box.data[2].item() < 10 or box.data[1].item() < 10 or box.data[0].item() > img.size[0]-10 or box.data[3].item() > img.size[1]-10:
            print("box is too close too border")
            # if len(boxes_orig.size()) > 1:
            #     draw2 = ImageDraw.Draw(img_orig)

            #     for box1 in boxes_orig:
            #         draw2.rectangle(list(box1), outline='red')
            #     img_orig.show()
            #     import pylab
            #     pylab.imshow(img_orig)
            #     pylab.show()
            # box.data[2].item() < 5
            # box.data[1].item() < 5
            # box.data[0].item() > img.size[0]
            # box.data[3].item() > img.size[1]
            
            # ratio = round(length/height,2)
            # oratio = round(olength/oheight,2)
            # if ratio != oratio:    
            #     print("old length height ratio:", olength/oheight)
            #     print("new length height ratio:", length/height)
            return img_orig, boxes_orig

    return img, boxes

def blur(img):
    if random.randint(0,100) < 50:
        img = img.filter(ImageFilter.GaussianBlur(radius = 1))
    return(img)

def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).

    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    if len(boxes):
        boxes -= torch.Tensor([j,i,j,i])
        boxes[:,0::2].clamp_(min=0, max=ow-1)
        boxes[:,1::2].clamp_(min=0, max=oh-1)
    return img, boxes

def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.

    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes):
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
    return img, boxes

def draw(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()


def test():
    img = Image.open('./image/000001.jpg')
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    img, boxes = random_crop(img, boxes)
    print(img.size)
    draw(img, boxes)

# test()
