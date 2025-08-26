import math
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_channels, base_channel, upscale_factor, num_residuals):
        super(Net, self).__init__()

        self.input_conv = nn.Conv2d(num_channels, base_channel, kernel_size=3, stride=1, padding=1)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(PixelShuffleBlock(base_channel, upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)

        self.output_conv = nn.Conv2d(base_channel, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x = torch.add(x, residual)
        x = self.upscale_layers(x)
        x = self.output_conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x * 0.1
        x = torch.add(x, residual)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel * (upscale_factor ** 2), kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x