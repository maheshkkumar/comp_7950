import torchvision
from torch import nn


class LocalizationNetwork(nn.Module):
    def __init__(self, epoch=0, pre_trained=None):
        super(LocalizationNetwork, self).__init__()
        self.pre_trained = pre_trained
        self.resent = torchvision.models.resnet50(pretrained=True)

        # logic to check for pre-trained weights from earlier checkpoint
        if pre_trained is not None:
            pass
        else:
            pass

    def forward(self, x):
        output = self.resent(x)

        return output
