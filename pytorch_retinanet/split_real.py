# script for splitting Benji's images into train and test groups

            
with open("../../Data/kuzikus_patches_800x600/labels/00_SPLIT.txt","r") as f:
    with open("../../Data/kuzikus_patches_800x600/labels/train.txt","w") as train:
        with open("../../Data/kuzikus_patches_800x600/labels/test.txt","w") as test:
            x=0
            for line in f:
                x+=1
                img_name = line.strip().split()[0]
                txt_name = img_name[:-3] + "txt"
                
                # if the image has a bounding box file, write the bounding box(es) to the train or test file
                try:
                    with open("../../Data/kuzikus_patches_800x600/labels/" + txt_name,"r") as box:
                        boxes = ''
                        for line in box:
                            box_values = line.strip().split()
                            boxes = boxes + ' ' + ' '.join(box_values[1:5]) + ' 1'
                        print(img_name + boxes)
                        if x<1001:
                            train.write(img_name + boxes + '\n')
                        if 1000<x<1501:
                            test.write(img_name + boxes + '\n')
                # if the image does not have animals/annotations, just write the file name in the train or test file
                except:
                    if x<1001:
                        train.write(img_name + '\n')
                    if 1000<x<1501:
                        test.write(img_name + '\n')

                #print(img_name)
