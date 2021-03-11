import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(ConvBlock, self).__init__()
        features = [in_channels] + [out_channels for i in range(layers)]
        convs = []
        for i in range(layers):
            convs.append(nn.Conv2d(in_channels = features[i], out_channels = features[i + 1], kernel_size = 3,
                                    padding = 1, bias = True))
            convs.append(nn.BatchNorm2d(num_features = features[i + 1], affine = True, track_running_stats = True))
            convs.append(nn.ReLU())

        self.op = nn.Sequential(*convs)

    def forward(self, x):
        return self.op(x)


