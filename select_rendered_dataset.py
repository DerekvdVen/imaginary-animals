
import argparse


# ##### DON"T RUN THIS AGAIN
# import random
# with open('../Data/labels/train_23456m_np.txt','r') as source:
#     data = [ (random.random(), line) for line in source ]
# data.sort()
# with open('../Data/labels/train_23456m_np_mixed.txt','w') as target:
#     for _, line in data:
#         target.write( line )
# #####

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')

parser.add_argument('-nim', default=5000, type=int, help = 'number of training images') # set default to 20304060m
args = parser.parse_args()
print("args: ", args)

amount_of_images = args.nim

with open("../Data/labels/train_23456m_np_mixed.txt","r") as input_file:
    with open("../Data/labels/train_23456m_np_mixed_" + str(amount_of_images) + ".txt","w") as output_file:
        count = 0
        count_animals = 0
        count_empty = 0
        for line in input_file:
            if count >= amount_of_images:
                break
            try:
                line.strip().split()[1]
                count_animals+=1
            except:
                count_empty+=1
                pass
            output_file.write(line)
            count+=1
            print(line)


print("n animals: " + str(count_animals))
print("n empty images: " + str(count_empty))
