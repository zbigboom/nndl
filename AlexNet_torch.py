import torch
from torch import nn
from torch.utils import data
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb')as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    train_path = 'E:\python\cifar10\data_batch_1'
    train_data = []
    train_labels = []
    train_data_tmp = unpickle(train_path)[b'data']
    for item in train_data_tmp:
        train_data.append(item)
    train_labels += unpickle(train_path)[b'labels']
    train_set_data = np.array(train_data)
    train_set_labels = np.array(train_labels)
    # train_set_labels=train_set_labels.reshape((1,train_set_labels.shape[0]))

    return train_set_data, train_set_labels


A, B = load_train_data()
print(A)


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, ndex):
        data = self.data[ndex]
        lablels = self.label[ndex]
        return data, lablels

    def __len__(self):
        return len(self.data)


train_set = GetLoader(A, B)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True, drop_last=False)


# 查看每个batch的数据及标签
# for i,data in enumerate(data):
#     print("NO{}batch\n{}".format(i,data))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU()
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU()
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                   )
        self.dense = nn.Sequential(nn.Linear(128, 120),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(120, 84),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(84, 10)
                                   )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)


def train():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    iter = 0
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            iter = iter + 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


train()
print('!!!Finish!!!')
