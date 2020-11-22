import torch
from torch import nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=2)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=2)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=2)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2)
        self.conv7=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv8=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.conv9=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.conv10=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv11=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.conv12=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.conv13=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.fc1=nn.Linear(512*7*7,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,1000)

    def forward(self,x):

        in_size=x.size(0)

        out=self.conv1(x)
        out=F.relu(out)
        out=self.conv2(out)
        out=F.relu(out)
        out=self.pool1(out)
        out=self.conv3(out)
        out=F.relu(out)
        out=self.conv4(out)
        out=F.relu(out)
        out=self.pool2(out)
        out=self.conv5(out)
        out=F.relu(out)
        out=self.conv6(out)
        out=F.relu(out)
        out=self.conv7(out)
        out=F.relu(out)
        out=self.pool3(out)
        out=self.conv8(out)
        out=F.relu(out)
        out=self.conv9(out)
        out=F.relu(out)
        out=self.conv10(out)
        out=F.relu(out)
        out=self.pool4(out)
        out=self.conv11(out)
        out=F.relu(out)
        out=self.conv12(out)
        out=F.relu(out)
        out=self.conv13(out)
        out=F.relu(out)
        out=self.pool5(out)

        out=out.view(in_size,-1)

        out=self.fc1(out)
        out=F.relu(out)
        out=nn.Dropout(0.5)

        out=self.fc2(out)
        out=F.relu(out)
        out=nn.Dropout(0.5)

        out=self.fc3(out)
        out=F.relu(out)
        out=nn.Dropout(0.5)

        out=F.log_softmax(out,dim=1)

        return out


