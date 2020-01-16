import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from Kuzikus_bigImageValidation import evalOnBigTensor
import os

#imports
checkpoint = "60m_ckpt10"

#dir = "/mnt/guanabana/raid/data/datasets/Kuzikus/SAVMAP/data/raster/ebee/2014-05/20140515_11_rgb/img/"
outdir = '../../output/output_images/' + checkpoint
dir = "../../Data/real/"
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



print('Loading image..')
for img_x in os.listdir(dir):
    image = Image.open(dir + img_x).convert('RGB')

    # for real image test
    # image size = 4000 3000
    # left upper right lower
    area = (3600, 1100, 4000, 1500)
    image = image.crop(area)
    
    w = h = 600
    image = image.resize((w,h))
    print(image)

    print('Predicting..')
    x = transform(image)
    x = x.unsqueeze(0)
    x = Variable(x, requires_grad = False)
    
    #loc_preds, cls_preds = evalOnBigTensor(net, x, 512, 2)

    loc_preds, cls_preds = net(x) # remove, for big images

    print('Decoding..')
    encoder = DataEncoder()
    print(loc_preds.data.squeeze())
    boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))
    # calculate statistics from these predicted boxes and labels, and the ground truth boxes and labels
    
    draw = ImageDraw.Draw(image)

    print(boxes)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    image.show()
    image.save(outdir + 'image_' + img_x  )

    import pylab
    pylab.imshow(image)
    pylab.show()
    
    print("plot the thing")