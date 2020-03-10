# script for splitting Benji's images into train and test groups
import argparse

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('-ni', default=250, type=int, help='number of images to put in train_x for finetuning') # set default to 20304060m
parser.add_argument('-p', default=10, type=float,help='percentage animals in image')
args = parser.parse_args()
print("args: ", args)

n_imgs = args.ni
perc = 100/args.p
print(n_imgs)
width = 800
height = 600
count=0
count_empty = 0
with open("../../Data/kuzikus_patches_800x600/labels/00_SPLIT.txt","r") as f:
    with open("../../Data/kuzikus_patches_800x600/labels/train" + str(n_imgs) + ".txt","w") as train:
        for line in f:
            
            img_name = line.strip().split()[0]
            txt_name = img_name[:-3] + "txt"
            
            # if the image has a bounding box file, write the bounding box(es) to the train or test file
            try:
                if count > (n_imgs/perc-1):
                    raise Exception("enough animals in file")
                with open("../../Data/kuzikus_patches_800x600/labels/" + txt_name,"r") as box:
                    boxes = ''
                    for line in box:
                        box_values = line.strip().split()[1:5]

                        # resize boxes to img size
                        box_values[0] = str(float(box_values[0])*width)
                        box_values[1] = str(float(box_values[1])*height)
                        box_values[2] = str(float(box_values[2])*width)
                        box_values[3] = str(float(box_values[3])*height)
                        
                        
                        # append box values as annotations
                        boxes = boxes + ' ' + ' '.join(box_values) + ' 1'
                    print(img_name + boxes)
                    train.write(img_name + boxes + '\n')
                    count += 1
            # if the image does not have animals/annotations, just write the file name in the train or test file
            except:
                if count_empty < n_imgs/perc*(perc-1):
                    train.write(img_name + '\n')
                    count_empty += 1
                else:
                    pass


