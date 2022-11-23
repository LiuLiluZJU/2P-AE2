import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net.unet_model import UNet
import torchvision.models as models


class SCNet(nn.Module):
    def __init__(self, input_channel, num_labels, spatial_downsample=8):
        super(SCNet, self).__init__()
        self.input_channel = input_channel
        self.num_labels = num_labels
        base_channel = 64
        self.node = nn.Sequential(
            nn.Conv2d(input_channel, base_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.unet = UNet(base_channel, base_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(base_channel, num_labels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(spatial_downsample)
        self.sconv = nn.Sequential(
            nn.Conv2d(num_labels, base_channel, kernel_size=7, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, kernel_size=7, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, kernel_size=7, padding=3),
            nn.LeakyReLU(inplace=True)
        )
        self.end_node = nn.Sequential(
            nn.Conv2d(base_channel, num_labels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        # self.end_node = nn.Conv2d(base_channel, num_labels, kernel_size=7, padding=3)
        # nn.init.normal_(self.end_node.weight.data, std=0.001)
        # self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=spatial_downsample, mode='bilinear', align_corners=True)
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.resnet18.fc = nn.Linear(512, 25)

    def forward(self, x):
        x1 = self.node(x)
        x2 = self.unet(x1)
        local_heatmaps = self.conv(x2)
        x3 = self.avg_pool(local_heatmaps)
        x4 = self.sconv(x3)
        x5 = self.end_node(x4)
        spatial_heatmaps = self.upsample(x5)

        heatmaps = local_heatmaps * spatial_heatmaps
        sigmas_uncertainty = self.resnet18(spatial_heatmaps)

        # return heatmaps, local_heatmaps, spatial_heatmaps, sigmas_uncertainty
        return heatmaps, local_heatmaps, spatial_heatmaps


