import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model_Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        ##resnet layer numbers of each block
        layer_num = [2, 2, 2, 2]
        
        self.layers = nn.ModuleDict()
        
        self.layers.add_module('conv0', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding="same"))
        self.layers.add_module('norm0', nn.BatchNorm2d(64))
        self.layers.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2))
        
        start = 1
        self.block1 = self._create_block(64, 64, 3, 1, 1, downsample=False, start_num=1, layer_num=layer_num[0])
        self.block2 = self._create_block(64, 128, 3, 2, 1, downsample=True, start_num=1+sum(layer_num[0:1]), layer_num=layer_num[1])
        self.block3 = self._create_block(128, 256, 3, 2, 1, downsample=True, start_num=1+sum(layer_num[0:2]), layer_num=layer_num[2])
        self.block4 = self._create_block(256, 512, 3, 2, 1, downsample=True, start_num=1+sum(layer_num[0:3]), layer_num=layer_num[3])
        
        self.layers.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.add_module('fc', nn.Linear(512, num_classes))
        
    def _create_block(self, in_channel, out_channel, kernel, stride, padding, downsample, start_num, layer_num):
        
        """
        Generate Resnet block from ResnetLayer
        in_channel: number of input data channels
        out_channel: number of output data channels
        kernel: kernel size
        stride : stride
        padding : padding
        downsample (Boolean): downsample should be applied or not 
        start_num: layer starting number. e.g. layer starts from 3 in case that two layers have been created previously
        layer_num: total layers in this block
        """
        self.layers.add_module("layer%d" % start_num, _ResnetLayer(in_channel, out_channel, kernel, stride, padding, downsample=True))
        for i in range(1, layer_num):
            start_num += i 
            self.layers.add_module("layer%d" % start_num, _ResnetLayer(out_channel, out_channel, kernel))
        
        
    def forward(self,x):
        for name, layer in self.layers.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = layer(x)
            #print(name, x.shape)
        return x
        
class _ResnetLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channel)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
    
    def forward(self, x):
        identity = x
        x = F.relu(self.norm(self.conv1(x)))
        x = self.norm(self.conv2(x))
        
        if(self.downsample):
            identity = self.downsample(identity)
            
        return F.relu(torch.add(identity, x))
        
if __name__ == "__main__":

    model_res18 = Model_Resnet18(10)
    for child in model_res18.children():
        print(child)

