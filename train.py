import argparse

import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from architecture.network import LocalizationNetwork
from dataloader.data_utils import read_file, compute_acc, box_transform, xywh_to_x1y1x2y2
from dataloader.dataloader import CUBDataset

# seeding random variable to reproduce results
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InitiateTraining(object):
    def __init__(self, args):
        self.train_path = args.train_images_path
        self.label_path = args.train_labels_path
        self.split_rate = args.split_rate
        self.pre_trained = args.pre_trained
        self.config = args.config
        self.epoch = args.epoch
        self.localization_model = LocalizationNetwork(pre_trained=self.pre_trained, epoch=self.epoch)
        self.optimizer = optim.Adam(self.localization_model.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.hyperparameters = read_file(self.config)
        self.best_epoch = 1e+10
        self.best_accuracy = 1e+10
        self.batch_size = 32

    def train(self):
        self.localization_model  # use cuda
        self.criterion  # use cuda
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        train_set = CUBDataset(self.train_path, self.label_path, mode='train', split_rate=self.split_rate,
                               transform=self.transform)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        validation_set = CUBDataset(self.train_path, self.label_path, mode='validation', split_rate=self.split_rate,
                                    transform=self.transform)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=True)

        epochs = 10
        print("==> Starting Training")
        for idx, train_epoch in enumerate(range(epochs)):

            total_accuracy, total_loss = 0.0, 0.0
            self.lr_scheduler.step()

            # training the localization network
            for batch_idx, data in enumerate(train_loader):
                self.localization_model.train()
                img, label, img_size = data
                label = box_transform(xywh_to_x1y1x2y2(label), img_size)

                _input = Variable(img)  # use cuda(device)
                _target = Variable(label)  # use cuda

                # resetting optimizer to not remove old gradients
                self.optimizer.zero_grad()

                # forward pass
                output = self.localization_model(_input)

                # backward pass
                loss = self.criterion(output, _target)
                loss.backward()
                self.optimizer.step()

                # compute accuracy for the prediction
                accuracy = compute_acc(output.data.cpu(), _target.data.cpu(), img_size)

                print(
                    "Epoch: {}/{}, Batch: {}/{}, Training Accuracy: {:3f}, Training Loss: {:3f}".format(idx + 1, epochs,
                                                                                                        batch_idx + 1,
                                                                                                        len(
                                                                                                            train_loader),
                                                                                                        accuracy, loss))

                total_accuracy += accuracy
                total_loss += loss

            total_loss = float(total_loss) / len(train_loader)
            total_accuracy = float(total_accuracy) / len(train_loader)

            val_accuracy, val_loss = 0.0, 0.0
            for batch_idx, data in enumerate(validation_loader):
                self.localization_model.train(False)
                img, label, img_size = data
                label = box_transform(xywh_to_x1y1x2y2(label), img_size)

                _input = Variable(img)  # use cuda
                _target = Variable(label)  # use cuda

                with torch.no_grad():
                    output = self.localization_model(_input)

                loss = self.criterion(output, _target)
                accuracy = compute_acc(output.data.cpu(), _target.data.cpu(), img_size)

                val_accuracy += accuracy
                val_loss += loss

            val_loss = float(val_loss) / len(validation_loader)
            val_accuracy = float(val_accuracy) / len(validation_loader)

            print(
                "Epoch: {}/{}, Training Accuracy: {:3f}, Training Loss: {:3f}, Validation Accuracy: {:3f}, Validation Loss: {:3f}".format(
                    idx + 1, epochs, total_accuracy, total_loss, val_accuracy, val_loss))

            if val_accuracy < self.best_accuracy:
                self.best_epoch = train_epoch + 1
                print("=> Best Epoch: {}, Accuracy: {:3f}".format(self.best_epoch, val_accuracy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tri', '--train_images_path', help='Path of the training image list',
                        default='./dataset/train_images.txt')
    parser.add_argument('-trl', '--train_labels_path', help='Path of the training labels list',
                        default='./dataset/train_boxes.txt')
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
