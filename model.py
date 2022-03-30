import torch
import torch.nn as nn

create_conv_layers = lambda in_channels, num_blocks: nn.Sequential(
    *[CNNBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1) for i in
      range(num_blocks)])


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyRelu(self.bn(self.conv(x)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, num_of_blocks):
        super(ResnetBlock, self).__init__()
        self.layers = create_conv_layers(in_channels, num_of_blocks)

    def forward(self, x):
        y = self.layers(x)
        x += y
        return x


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()


class MyResModel(torch.nn.Module):
    def __init__(self, num_of_classes):
        super(MyResModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_of_classes),
        )

    def forward(self, x):
        return self.classifier(x)

