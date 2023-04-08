import os
from cv2 import imread
import numpy as np

CHUNK = 20000
TRAINSIZE = 66811
TESTSIZE = 16703
np.random.seed(1)

def dataset_reader(train: bool, chunk=0):
    """ A function that read training dataset or validation dataset
        from ../data
    
    Args:
        train_or_val: boolean, which indicates the user ask for training dataset or validation dataset
            True for training dataset, False for validation dataset
    
    Return:
        dataset: (what type???)
    """
    if(train == True):
        PATH = "../data/train"  #path to your training image
        dic = np.load("../data/y_train.npy", allow_pickle=True).item()
    else:
        PATH = "../data/test"  #path to your testing image
        dic = np.load("../data/y_valid.npy", allow_pickle=True).item()

    file_dir = os.listdir(PATH) #read the images from the directory
    np.random.shuffle(file_dir)
    file_dir = file_dir[CHUNK*chunk: min(len(file_dir), CHUNK*(chunk + 1))]

    cnt = 0
    data = []
    label = []
    for image in file_dir:
        y_label = dic[image]
        img= imread(PATH+'/'+image)
        data.append(img)
        label.append(y_label)
        cnt += 1
        if cnt % 1000 == 0:
            print("{}: {} => {}".format(cnt, image, y_label))
    data = np.array(data)
    label = np.array(label)

    return data, label

# Read training dataset
for chunk in range(0, int(np.ceil(TESTSIZE/CHUNK)), 1):
    print("Truck {}:".format(chunk))
    train_x, train_y = dataset_reader(train = False, chunk=chunk)

    np.save("./test_x_chunk{}.npy".format(chunk), train_x)
    np.save("./test_y_chunk{}.npy".format(chunk), train_y)