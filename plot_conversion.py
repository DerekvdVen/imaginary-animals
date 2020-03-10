#%matplotlib inline
import matplotlib.pyplot as plt

trainlist = []
testlist = []
with open("../output/graphs_lists/20304060m_250lr_train_losses.txt","r") as file:
    for item in file:
        trainlist.append(float(item.strip()))
with open("../output/graphs_lists/20304060m_250lr_train_losses.txt","r") as file:
    for item in file:
        testlist.append(float(item.strip()))

plt.plot(trainlist,c="red")
plt.title("train loss")
plt.xlabel("epochs")
plt.show()

plt.plot(testlist)
plt.title("test loss")
plt.xlabel("epochs")
plt.show()

min_test = min(testlist)
min_test_epoch = testlist.index(min_test)
min_train = trainlist[min_test_epoch]
print("epochs untill lowest test error: ", min_test_epoch)
print("minimum test error: ", min_test)
print("train error at minimum test error: ", min_train)