import argparse
import math
import os

import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.data_utils import *
from dataloader.dataloader import CUBDataset

# seeding random variable to reproduce results
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InitiateTesting(object):
    def __init__(self, args):
        self.test_path = args.test_images_path
        self.pre_trained = args.pre_trained
        self.config = args.config
        self.save_output = args.save_output
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.hyperparameters = read_file(self.config)
        self.batch_size = self.hyperparameters['batch_size']

        if not os.path.exists(self.save_output):
            os.makedirs(self.save_output)

        self.save_output = os.path.join(args.save_output, 'results.txt')

    def test(self):
        output_boxes = []

        model = torchvision.models.resnet18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 4)

        # logic to check for pre-trained weights from earlier checkpoint
        if self.pre_trained is not None:
            pre_trained_model = torch.load(self.pre_trained)
            model.load_state_dict(pre_trained_model)

        model.to(device)  # use cuda

        test_set = CUBDataset(self.test_path, mode='test', transform=self.transform)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        print("==> Starting Testing")

        # testing the localization network
        model.eval()

        for batch_idx, data in enumerate(test_loader):
            img, img_size = data

            _input = Variable(img.to(device))  # use cuda

            with torch.no_grad():
                output = model(_input)

            box_coordinates = box_transform_inv(output.data.cpu(), img_size)
            xywh = x1y1x2y2_to_xywh(box_coordinates)
            # output_boxes.append(xywh.data.numpy())
            for value in xywh.data.numpy():
                print(type(value))
                value = map(math.ceil, value.tolist())
                print(value)
                output_boxes.append(value)

        with open(self.save_output, 'w') as f:
            for item in output_boxes:
                f.write("{}\n".format(" ".join(str(i) for i in item)))

        print("Batch progress: {}/{}".format(batch_idx + 1, len(test_loader)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tei', '--test_images_path', help='Path of the testing image list',
                        default='./dataset/test_images.txt')
    parser.add_argument('-pre', '--pre_trained', help='Path of the pre-trained checkpoint', default=None)
    parser.add_argument('-c', '--config', help='Path of the hyperparameter configuration file',
                        default='./configuration/config.json')
    parser.add_argument('-so', '--save_output', help='Path for saving the test output',
                        default='./results')

    args = parser.parse_args()
    test_localization = InitiateTesting(args)
    test_localization.test()


if __name__ == '__main__':
    main()
