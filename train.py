import argparse

import torch
from torchvision import transforms

from architecture.network import LocalizationNetwork
from scaffolding.utils import read_file


class InitiateTraining(object):
    def __init__(self, args):
        self.train_path = args.train_images_path
        self.label_path = args.train_labels_path
        self.split_rate = args.split_rate
        self.pre_trained = args.pre_trained
        self.config = args.config
        self.epoch = args.epoch
        self.localization_model = LocalizationNetwork(pre_trained=self.pre_trained, epoch=self.epoch)
        # self.optimizer = torch.optim.Adam()
        self.transforms = transforms.Compose([transforms.ToTensor, transforms.Resize(224)])
        self.hyperparameters = read_file(self.config)

    def train(self):
        # self.localization_model.train()
        # train_set = CUBDataset(self.train_path, self.label_path, mode='train', split_rate=self.split_rate,
        #                        transform=self.transforms)
        # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        # validation_set = CUBDataset(self.train_path, self.label_path, mode='validation', split_rate=self.split_rate,
        #                             transform=self.transforms)
        # validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)

        # start training
        # compute accuracy on the validation set
        # save best model via checkpoint
        # implement adjust learning rate

        print(self.hyperparameters)
        print(self.localization_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tri', '--train_images_path', help='Path of the training image list', default='./dataset/train_images.txt')
    parser.add_argument('-trl', '--train_labels_path', help='Path of the training labels list', default='./dataset/train_boxes.txt')
    parser.add_argument('-s', '--split_rate', help='Split rate for training vs. validation', default=0.8)
    parser.add_argument('-pre', '--pre_trained', help='Path of the pre-trained checkpoint', default=None)
    parser.add_argument('-c', '--config', help='Path of the hyperparameter configuration file',
                        default='./configuration/config.json')
    parser.add_argument('-e', '--epoch', help='Epoch value for start from the checkpoint', default=0)

    args = parser.parse_args()
    model = InitiateTraining(args)
    model.train()


if __name__ == '__main__':
    main()
