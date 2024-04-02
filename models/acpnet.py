import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ["acpnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001))
            # ('bn', nn.BatchNorm2d(out_channels))
        ]))


class SelfAttention(nn.Module):
    r"""
    A self-attention module.
    """
    def __init__(self, in_channels, key_channels, value_channels):
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        
        self.key_conv = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1)
        self.query_conv = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.value_channels, self.in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        
        # count query、key、value
        query = self.query_conv(inputs)
        query = torch.reshape(query, (batch_size, self.key_channels, -1))

        key = self.key_conv(inputs)
        key = torch.reshape(key, (batch_size, self.key_channels, -1))

        value = self.value_conv(inputs)
        value = torch.reshape(value, (batch_size, self.key_channels, -1))

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)
        
        out = torch.bmm( value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.value_channels, height, width)
        out = self.out_conv(out)
        
        out = self.gamma * out + inputs
        
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // self.reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.size()
        y = self.squeeze(inputs).view(b, c)
        y = self.excitation(y)
        y = torch.reshape(y, [b, c, 1, 1])
        out = inputs * y.expand_as(inputs)
        return out


class ARBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, down_sample=False, self_attention=False, channel_attention=True):
        super(ARBlock, self).__init__()
        self.__in_channel = in_channel
        self.__out_channel = out_channel
        self.__down_sample = down_sample
        self.__self_attention = self_attention
        self.__channel_attention = channel_attention
        self.__strides = [2, 1] if down_sample else [1, 1]
        self.__kernel_size = (3, 3)
        
        self.conv_bn0 = ConvBN(self.__in_channel, self.__out_channel, kernel_size=self.__kernel_size, stride=self.__strides[0])
        self.conv_bn1 = ConvBN(self.__out_channel, self.__out_channel, kernel_size=self.__kernel_size, stride=self.__strides[1])
        self.relu = nn.ReLU(inplace=True)

        if self.__self_attention == True:
            self.sa_block = SelfAttention(in_channels=self.__out_channel, key_channels=self.__out_channel // 4, value_channels=self.__out_channel // 4)
        
        if self.__channel_attention == True:
            self.ca_block = ChannelAttention(in_channels=self.__out_channel)

        if self.__down_sample == True:
            self.res_conv_bn = ConvBN(self.__in_channel, self.__out_channel, kernel_size=(1, 1), stride=2)

    def forward(self, inputs):
        res = inputs

        x = self.conv_bn0(inputs)
        x = self.relu(x)
        x = self.conv_bn1(x)

        if self.__self_attention == True:
            x = self.sa_block(x)

        if self.__channel_attention == True:
            x = self.ca_block(x)

        if self.__down_sample == True:
            res = self.res_conv_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = x + res
        out = self.relu(x)
        return out


class ACPNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ACPNet, self).__init__()
        self.conv_bn0 = ConvBN(2, 32, kernel_size=(1, 7), stride=(1, 3))
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 4))

        self.ARBlock_0_0 = ARBlock(32, 32)
        self.ARBlock_0_1 = ARBlock(32, 32)

        self.ARBlock_1_0 = ARBlock(32, 64, down_sample=True)
        self.ARBlock_1_1 = ARBlock(64, 64)

        self.ARBlock_2_0 = ARBlock(64, 128, down_sample=True)
        self.ARBlock_2_1 = ARBlock(128, 128)

        self.ARBlock_3_0 = ARBlock(128, 256, down_sample=True)
        self.ARBlock_3_1 = ARBlock(256, 256)

        self.avg_pool = nn.AvgPool2d([2, 10])
        self.fc0 = nn.Linear(256, 1000)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.001, inplace=True)
        self.fc1 = nn.Linear(1000, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        x = self.conv_bn0(inputs)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.ARBlock_0_0(x)
        x = self.ARBlock_0_1(x)
        x = self.ARBlock_1_0(x)
        x = self.ARBlock_1_1(x)
        x = self.ARBlock_2_0(x)
        x = self.ARBlock_2_1(x)
        x = self.ARBlock_3_0(x)
        x = self.ARBlock_3_1(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.leakyrelu(x)
        x = self.fc1(x)
        return x


def acpnet(num_classes=3):
    r""" Create a proposed ACPNet.

    :param num_classes
    :return: an instance of ACPNet
    """

    model = ACPNet(num_classes)
    return model