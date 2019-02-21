from torch import nn


class LocalizationNetwork(nn.Module):
    """
    Object localization network implementation using pre-trained ResNet18
    """

    def __init__(self, pre_trained=None):
        super(LocalizationNetwork, self).__init__()
        self.pre_trained = pre_trained

    def forward(self, x):
        output = self.model(x)

        return output
