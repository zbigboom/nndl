import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
                                 )
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
                                 )
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU()
                                 )
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU()
                                 )
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
                                 )
        self.dense=nn.Sequential(nn.Linear(9216,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,1000)
                                 )
    def forward(self,x):
        conv1_out=self.conv1(x)
        conv2_out=self.conv2(conv1_out)
        conv3_out=self.conv3(conv2_out)
        conv4_out=self.conv4(conv3_out)
        conv5_out=self.conv5(conv4_out)
        res=conv5_out.view(conv5_out.size(0),-1)
        out=self.dense(res)
        return out