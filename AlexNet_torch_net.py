import torch
from torch import nn


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # 卷积层1-两个大小为11*11*3*28的卷积核步长为4零填充3得到两个55*55*48的特征映射矩阵
        # 汇聚层1-大小3*3步长为2得到两个27*27*128的特征映射矩阵
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
                                   )
        # 卷积层2-两个大小为5*5*48*128的卷积核步长为1零填充2得到两个27*27*128的特征映射矩阵
        # 汇聚层2-大小3*3步长为2得到两个13*13*128的特征映射矩阵
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
                                   )
        # 卷积层3-两个路径的融合一个3*3*256*384的卷积核步长为1零填充1得到两个13*13*192的特征映射矩阵
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU()
                                   )
        # 卷积层4-两个大小为3*3*192*192的卷积核步长1零填充1得到两个13*13*192的特征映射矩阵
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU()
                                   )
        # 卷积层5-两个大小为3*3*192*128的卷积核步长1零填充1得到两个13*13*128的特征映射矩阵
        # 汇聚层3-大小为3*3步长为2得到两个大小为6*6*128的特征映射矩阵
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
                                   )
        # 三个全连接层大小为4096，4094，1000，使用relu激活函数dropout0.5
        self.dense = nn.Sequential(nn.Linear(9216, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4096, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4096, 1000)
                                   )

    def forward(self, x):
        # 定义前馈网络
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        # 将数据拉直
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out
