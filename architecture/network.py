import torchvision
from torch import nn


class LocalizationNetwork(nn.Module):
    """
    Object localization network implementation using pre-trained ResNet50
    """

    def __init__(self, epoch=0, pre_trained=None):
        super(LocalizationNetwork, self).__init__()
        self.pre_trained = pre_trained
        self.resnet = torchvision.models.resnet50(pretrained=True)

        # logic to check for pre-trained weights from earlier checkpoint
        if pre_trained is not None:
            pass
        else:
            fc_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(fc_features, 4)

    def forward(self, x):
        output = self.resnet(x)

        return output
