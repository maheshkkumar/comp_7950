import torch
import torchvision
from torch import nn


class LocalizationNetwork(nn.Module):
    """
    Object localization network implementation using pre-trained ResNet50
    """

    def __init__(self, epoch=0, pre_trained=None):
        super(LocalizationNetwork, self).__init__()
        self.pre_trained = pre_trained


    def forward(self, x):
        output = self.model(x)

        return output
