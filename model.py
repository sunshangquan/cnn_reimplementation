import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
#https://blog.csdn.net/sunqiande88/article/details/80100891
class resblock(nn.Sequential):
    def __init__(self, in_channels, hidden_channel=64, bottleneck=False):
        super().__init__()
        self.bottleneck = bottleneck
        stride = 2 if bottleneck else 1
        self.sub_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel)
        )
        if self.bottleneck:
            self.conv2 = nn.Conv2d(in_channels, hidden_channel, 1, stride=stride, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channel)
        
    def forward(self, x):
        out = self.sub_block(x)
        if self.bottleneck:
            shortcut = self.bn2(self.conv2(x))
            return F.relu(out + shortcut)
        return F.relu(out + x)
        

class resnet(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        channels = [64]*3 + [128]*2# + [256]*2 + [512]*2
        id_bottleneck = [2, 4, 6]
        self.resblocks = nn.Sequential(OrderedDict(
                    [("res_block{}".format(i), resblock(channels[i], channels[i+1], (i in id_bottleneck))) for i in range(len(channels)-1)
                    ]))
        
        self.pool0 = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(128, n_class)
        # self.softmax = nn.Softmax()
        
    def forward(self, x):
        # return super().forward(x)
        x = self.layer0(x)
        x = self.resblocks(x)

        # for i, block in enumerate(self.resblocks):
        #     x = block(x)
        x = self.pool0(x)
        x = self.fc0(x.squeeze())
        # x = self.softmax(x)
        return x
