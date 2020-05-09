img_count=0
animal_count=0
with open("../Data/labels/train_23456m_np_mixed_10000.txt","r") as f:
    for line in f:
        if len(line.split()) > 1:
            print(line)
            print((len(line.split())-1)/5)
            img_count=img_count+1
            animal_count = animal_count + (len(line.split())-1)/5

print("n images ",img_count)
print("n animals", animal_count)