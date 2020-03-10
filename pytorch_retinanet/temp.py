dir = "../../Data/kuzikus_patches_800x600/images/"
val_loc = "../../Data/kuzikus_patches_800x600/labels/train.txt" # test on smaller things
# checkpoint = "20304060m"
# dir = '../../Data/images/val/' # this will be where the val images are
# val_loc = "../../Data/labels/val_" + checkpoint + '.txt'

size = w = h = 600
origsize = origw = origh = 512
changeboxtypebool = True
with open(val_loc) as annotations_file: # this will be where the val.txt file is. 
    for line in annotations_file:
        # count = count+1
        # if count > 30:
        #     break

        line_list = line.strip().split(' ')
        img_x = line_list[0]
        
        boxesdata = line_list[1:]
        nboxes = int(len(boxesdata)/5)
        loc_targets = []
        loc_targets_new=[]
        cls_targets = []

        # recast loc_targets to the new size image
        for i in range(0,len(boxesdata),5):
            loc_targets.append([round(float(j)/origsize*size,4) for j in boxesdata[i:i+4]])
            box = []

            if changeboxtypebool == True:
                centerx = float(boxesdata[0])
                centery = float(boxesdata[1])
                boxwidth = float(boxesdata[2])
                boxheight = float(boxesdata[3])
                boxesdata[0] = centerx - boxwidth/2
                boxesdata[1] = centery - boxheight/2
                boxesdata[2] = centerx + boxwidth/2
                boxesdata[3] = centery + boxheight/2

            box.append(round(float(boxesdata[0])/origw*w,4))
            box.append(round(float(boxesdata[1])/origh*h,4))
            box.append(round(float(boxesdata[2])/origw*w,4))
            box.append(round(float(boxesdata[3])/origh*h,4))
            loc_targets_new.append(box)

            print("test:", loc_targets)
            print("new: ", loc_targets_new)
            cls_targets.append(int(boxesdata[i+4]))
