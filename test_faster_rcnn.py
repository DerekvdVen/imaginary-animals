from PIL import Image
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGeneratorr
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T


# Define class
class Animals_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images_with_animals/2019-11"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks/2019-11"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images_with_animals/2019-11", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks/2019-11", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Get functions
def edit_model(model):

    # let's make the RPN generate 3 x 3 anchors per spatial location, with 3 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
 
    # let's define what are the feature maps that we will use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling. if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    return anchor_generator, roi_pooler
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# replace the classifier with a new one, get number of imput features for the classifier, and replace the pre-trained head with a new one
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 


# load a pre-trained model for classification and return only the features, FasterRCNN needs to know the n output channels in a backbone, mobilenet_v2 = 1280
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280


# load model with new backbone
# model = FasterRCNN(backbone,num_classes=2)

# fancy model
anchor_generator, roi_pooler = edit_model()
model = FasterRCNN(backbone,
                num_classes=2,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler)


# set dataset and defined transformations
dataset_train = Animals_dataset('../Data/', get_transform(train=True))
dataset_test = Animals_dataset('../Data/', get_transform(train=False))


# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# get the model using our helper function and move to device
model = get_instance_segmentation_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)


# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# load model and set to evalutation mode
PATH = '../models/faster_rcnn.pth'
model.load_state_dict(torch.load(PATH))
model.eval()



# test and save images
for x in range(10):
    img, _ = dataset_test[x]


    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    print(prediction)

    
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # plt.imshow(img)
    # plt.show()
    # im = Image.fromarray(img)

    img.save('../output/image_' + str(x) + ".png")
    try:
        img_result = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        #plt.imshow(img_result)
        #plt.show()
        #im_2 = Image.fromarray(img_result)
        img_result.save('../output/image_' + str(x) + 'seg.png')
    except:
        print("hmm I guess it didn't detect any animals?")

