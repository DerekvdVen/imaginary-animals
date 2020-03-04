import os

# check for animals takes images from Data/all/60m|30m and moves the animal images into Data/only_animal_images/all/60m|30m
print("check all images for animals")
#os.system("python check_for_animals.py")

# # takes input from Data/all/60m|30m and splits the data into train and test groups in Data/semantic|images/60|30m/ 
# print("creating training and validation sets")
# os.system(python split_train_test.py)

# # takes input from Data/semantic|images/60|30m/ and creates annotations in a train.txt and test.txt file 
# print("run main.py with train/test distributions")
# os.system(python main.py)

# # runs the CNN with data from Data/all/60|30m and annotations from train.txt and test.txt
# print("training retinanet")
os.system("cd pytorch_retinanet ")
os.system("pwd")
# os.system(python train.py)

# # check model output with test.py or val.py
# os.system(python test.py)