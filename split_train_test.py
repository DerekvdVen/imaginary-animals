import os
import re
import numpy as np
import statistics as st
from shutil import copyfile
import random
import argparse

#dist = "30m"
parser = argparse.ArgumentParser(description='split_train_test')
parser.add_argument('-dir', default="30m", type=str, help='directory named after distance: 30m, 60m, 3060m')
args = parser.parse_args()
dist = args.dir

print("args: ",args)

def get_test_train(dir,copydir,amount_of_stdv = 0.5):


    pattern = re.compile("(?<=m_)(.*?)(?=\.)")
    comma = re.compile(",")
    xlist = []
    test = []
    train = []
    val = []


    for image in os.listdir(dir):
        print(image)
        xcode = int(comma.split(pattern.findall(image)[0])[0])
        xlist.append(xcode)

    print("xlist: ", len(xlist))
    percent_95 = st.mean(xlist) - amount_of_stdv* st.stdev(xlist)
    percent_90 = st.mean(xlist) - amount_of_stdv* st.stdev(xlist) *1.5
    
    # maybe add another percent_90 for test and > percent_95 is then val

    for image in os.listdir(copydir):
        xcode = int(comma.split(pattern.findall(image)[0])[0])
        if xcode <= percent_95 and xcode >= percent_90:
            test.append(image)
            print("test",image)
        elif xcode <= percent_90:
            val.append(image)
            print("val",image)
        else:
            train.append(image)
            print("train",image)
    print("perc90:", percent_90)
    print("perc95:", percent_95)
    print("test percentage: ", len(test)/(len(test)+len(train)+len(val))*100)
    print("val percentage: ", len(val)/(len(test)+len(train)+len(val))*100)
    return(test,train,val)

def remove_copy(dir,segdir):
    #remove
    for x in ["images","semantic","only_animal_images"]:
        os.system("rm ../Data/" + x + "/test/*")
        os.system("rm ../Data/" + x + "/train/*")
        os.system("rm ../Data/" + x + "/val/*")
    print("temporary test and train data removed from dir")
    
    #copy test data
    for img in test:
        copyfile(dir + img, "../Data/images/test/" + img)
        copyfile(segdir + img, "../Data/semantic/test/" + img)
    print("testfiles copied")
    print(len(test))

    #copy train data
    for img in train:
        copyfile(dir + img, "../Data/images/train/" + img)
        copyfile(segdir + img, "../Data/semantic/train/" + img)

    print("trainfiles copied")
    print(len(train))

    #copy val data
    for img in val:
        copyfile(dir + img, "../Data/images/val/" + img)
        copyfile(segdir + img, "../Data/semantic/val/" + img)
    
    print("valfiles copied")
    print(len(val))

# Create training and test sets
test, train, val = get_test_train("../Data/only_animal_images/all/" + dist, "../Data/images/all/" + dist, amount_of_stdv = 1)

# Copy train and test datasets to correct locations
remove_copy("../Data/images/all/" + dist + "/","../Data/semantic/all/" + dist + "/")