import torch
import torch.nn as nn
from collections import OrderedDict
import thop
# from utils import logger, line_seg

line_seg = ''.join(['*'] * 65)


__all__ = ["chin"]


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn0', nn.BatchNorm2d(out_channels)),
            ('relu0', nn.ReLU()),   
            ('conv1', nn.Conv2d(out_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU())
            # ('bn', nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)),
            # ('bn', nn.BatchNorm2d(out_channels))
        ]))


class Chin(nn.Module):
    def __init__(self):
        super(Chin, self).__init__()

        self.conv_pool0 = ConvBN(2, 32, kernel_size=(1, 3), stride=1)
        self.avg_pool0 = nn.AdaptiveAvgPool2d((16, 232))
        self.conv_pool1 = ConvBN(32, 64, kernel_size=(1, 3), stride=1)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((16, 58))
        self.conv_pool2 = ConvBN(64, 128, kernel_size=(1, 3), stride=1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((16, 15))
        self.conv_pool3 = ConvBN(128, 256, kernel_size=(1, 3), stride=1)
        self.avg_pool3 = nn.AdaptiveAvgPool2d((16, 4))

        self.fc0 = nn.Linear(256, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.conv_pool0(inputs)
        out = self.avg_pool0(out)

        out = self.conv_pool1(out)
        out = self.avg_pool1(out)

        out = self.conv_pool2(out)
        out = self.avg_pool2(out)

        out = self.conv_pool3(out)
        out = self.avg_pool3(out)

        out = out.view(-1, 256)

        out = self.fc0(out)
        out = self.relu(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)
        return out

def chin():
    model = Chin()
    return model


model = chin()



image = torch.randn([1, 2, 16, 924])
flops, params = thop.profile(model, inputs=(image,), verbose=False)
flops, params = thop.clever_format([flops, params], "%.3f")

print(f'=> Model Flops: {flops}')
print(f'=> Model Params Num: {params}\n')
print(f'{line_seg}\n{model}\n{line_seg}\n')