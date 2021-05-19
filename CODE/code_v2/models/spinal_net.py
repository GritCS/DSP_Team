from .dec_net import DecNet
from . import resnet34
import torch.nn as nn
import torch
import numpy as np
class SpineNet(nn.Module):
    def __init__(self, heads, basename, pretrained, down_ratio, final_kernel, head_conv):
        super(SpineNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))

        self.base_network = eval(basename+'.get_pose_net')(pretrained=True)
        # 跟据basenet设置的dec_net的通道数
        if basename == 'resnet34':
            channels = [3, 64, 64, 128, 256, 512]  # resnet34
        elif basename == 'resnet152':
            channels = [3, 64, 256, 512, 1024, 2048]  # resnet152
        elif basename == 'resnet101':
            channels = [3, 64, 256, 512, 1024, 2048]  # resnet101

        elif basename == 'mobilenetv2':
            channels = [3, 32, 24, 32, 64, 320]  # mobilenetv2
        elif basename == 'shufflenetv2':
            channels = [3, 24, 24, 116, 232, 464] # shufflenetv2
        print(channels)
        self.dec_net = DecNet(heads, final_kernel, head_conv,channels, channels[self.l1])

    def get_feature_map(self,x):
        x = self.base_network(x)
        return self.dec_net.get_feature_map(x)

    def forward(self, x):
        x = self.base_network(x)
        dec_dict = self.dec_net(x)
        return dec_dict