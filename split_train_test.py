import os
import re
import numpy as np
import statistics as st
from shutil import copyfile


def get_test_train(dir,amount_of_stdv = 2):


    pattern = re.compile("(?<=\_)(.*?)(?=\.)")
    comma = re.compile(",")
    xlist = []
    test = []
    train = []


    for image in os.listdir(dir):
        print(image)
        xlist.append(int(comma.split(pattern.findall(image)[0])[0]))


    percent_95 = st.mean(xlist) + amount_of_stdv* st.stdev(xlist)


    for image in os.listdir(dir):
        if int(comma.split(pattern.findall(image)[0])[0]) >= percent_95:
            test.append(image)
        else:
            train.append(image)

    print(len(test)/(len(test)+len(train))*100)
    return(test,train)

# Create training and validation sets
test, train = get_test_train("../Data/images/all", amount_of_stdv = 1.5)


for img in test:
    copyfile("../Data/images/all/" + img, "../Data/images/val/" + img)
    copyfile("../Data/semantic/all/" + img, "../Data/semantic/val/" + img)
    

print("testfiles copied")

for img in train:
    copyfile("../Data/images/all/" + img, "../Data/images/train/" + img)
    copyfile("../Data/semantic/all/" + img, "../Data/semantic/train/" + img)

print("trainfiles copied")
