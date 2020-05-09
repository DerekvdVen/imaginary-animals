import os

val_loc = "../../Data/kuzikus_patches_800x600/labels/val.txt" # test on smaller things

count = 0
with open(val_loc,"r") as annotations_file: # this will be where the val.txt file is. 
    with open("../../Data/kuzikus_patches_800x600/labels/val_animals.txt","w") as ex:
        for line in annotations_file:
            linesplit = line.replace("/","-").split()
            #print(line[0])
            if linesplit[0] in os.listdir("../../output/output_images/23456z3_looksgood_0.1"):
                print(line)
                ex.write(line)