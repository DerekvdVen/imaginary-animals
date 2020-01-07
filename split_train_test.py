import os
import re
import numpy as np
import statistics as st
from shutil import copyfile

dist = "60m"

def get_test_train(dir,copydir,amount_of_stdv = 2):


    pattern = re.compile("(?<=\_)(.*?)(?=\.)")
    comma = re.compile(",")
    xlist = []
    test = []
    train = []


    for image in os.listdir(dir):
        print(image)
        xlist.append(int(comma.split(pattern.findall(image)[0])[0]))

    print("xlist: ", len(xlist))
    percent_95 = st.mean(xlist) - amount_of_stdv* st.stdev(xlist)


    for image in os.listdir(copydir):
        if int(comma.split(pattern.findall(image)[0])[0]) <= percent_95:
            test.append(image)
        else:
            train.append(image)

    print(len(test)/(len(test)+len(train))*100)
    return(test,train)

def remove_copy(dir,segdir):
    #remove
    for x in ["images","semantic","only_animal_images","masks"]:
        os.system("rm ../Data/" + x + "/test/*")
        os.system("rm ../Data/" + x + "/train/*")
    print("temporary test and train data removed from dir")
    
    #copy val data
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

# Create training and test sets
test, train = get_test_train("../Data/only_animal_images/all/" + dist, "../Data/images/all/" + dist, amount_of_stdv = 1)

# Copy train and test datasets to correct locations
remove_copy("../Data/images/all/" + dist + "/","../Data/semantic/all/" + dist + "/")