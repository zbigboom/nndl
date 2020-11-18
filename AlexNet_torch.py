import torch
import os
import torchvision
from torchvision import transforms
import numpy as np

def unpickle(file):
    import pickle
    with open(file,'rb')as fo:
        dict=pickle.load(fo,encoding='bytes')
    return dict
def load_train_data():
    train_path = 'E:\python\cifar10\data_batch_1'
    train_data = []
    train_labels = []
    train_data_tmp = unpickle(train_path)[b'data']
    for item in train_data_tmp:
        train_data.append(item)
    train_labels = unpickle(train_path)[b'labels']

    # train_data=np.array(train_data)
    # train_labels=np.array(train_labels)
    # train_labels=train_labels.reshape((1,train_labels.shape[0]))

    return train_labels,train_data
A,B=load_train_data()
print(B)


