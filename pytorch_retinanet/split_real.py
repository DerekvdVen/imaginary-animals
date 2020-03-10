# script for splitting Benji's images into train and test groups

n_train = 8000
n_test = 1500
n_val = 1500
width = 800
height = 600
with open("../../Data/kuzikus_patches_800x600/labels/00_SPLIT.txt","r") as f:
    with open("../../Data/kuzikus_patches_800x600/labels/train.txt","w") as train:
        with open("../../Data/kuzikus_patches_800x600/labels/test.txt","w") as test:
            with open("../../Data/kuzikus_patches_800x600/labels/val.txt","w") as val:
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
                                box_values = line.strip().split()[1:5]

                                # resize boxes to img size
                                box_values[0] = str(float(box_values[0])*width)
                                box_values[1] = str(float(box_values[1])*height)
                                box_values[2] = str(float(box_values[2])*width)
                                box_values[3] = str(float(box_values[3])*height)
                                
                                
                                # append box values as annotations
                                boxes = boxes + ' ' + ' '.join(box_values) + ' 1'
                            print(img_name + boxes)
                            if x < (n_train+1):
                                train.write(img_name + boxes + '\n')
                            if n_train < x < (n_train + n_test + 1):
                                test.write(img_name + boxes + '\n')
                            if (n_train + n_test + 1) < x < (n_train + n_test + n_val + 1):
                                val.write(img_name + boxes + '\n')
                                
                    # if the image does not have animals/annotations, just write the file name in the train or test file
                    except:
                        if x < (n_train+1):
                            train.write(img_name + '\n')
                        if n_train < x < (n_train + n_test + 1):
                            test.write(img_name + '\n')
                        if (n_train + n_test + 1) < x < (n_train + n_test + n_val + 1):
                            val.write(img_name + '\n')



