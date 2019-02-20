import argparse
import os

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

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
        self.experiment = args.experiment
        self.save_model = args.save_model
        self.criterion = nn.SmoothL1Loss()
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.hyperparameters = read_file(self.config)
        self.best_epoch = 1e+10
        self.best_accuracy = 1e+10
        self.batch_size = self.hyperparameters['batch_size']
        self.writer = SummaryWriter()

        if not os.path.exists(self.save_model):
            os.makedirs(self.save_model)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for parameter in m.parameters():
                parameter.requires_grad = False

    def train(self):

        model = torchvision.models.resnet18(pretrained=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # logic to check for pre-trained weights from earlier checkpoint
        if self.pre_trained is not None:
            pre_trained_model = torch.load(self.pre_trained)
            model.load_state_dict(pre_trained_model)
        else:
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, 4)

        model.to(device)  # use cuda
        self.criterion.to(device)  # use cuda
        self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_set = CUBDataset(self.train_path, label_path=self.label_path, mode='train', split_rate=self.split_rate,
                               transform=self.transform)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        validation_set = CUBDataset(self.train_path, label_path=self.label_path, mode='validation', split_rate=self.split_rate,
                                    transform=self.transform)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=True)

        epochs = 10
        print("==> Starting Training")
        for idx, train_epoch in enumerate(range(epochs)):

            total_accuracy, total_loss = 0.0, 0.0
            self.lr_scheduler.step()

            # training the localization network
            for batch_idx, data in enumerate(train_loader):
                model.train()

                # freezing batch normalization layers
                # self.localization_model.apply(self.set_bn_eval)

                img, label, img_size = data
                label = box_transform(xywh_to_x1y1x2y2(label), img_size)

                _input = Variable(img.to(device))  # use cuda(device)
                _target = Variable(label.to(device))  # use cuda

                # resetting optimizer to not remove old gradients
                optimizer.zero_grad()

                # forward pass
                output = model(_input)

                # backward pass
                loss = self.criterion(output, _target)
                loss.backward()
                optimizer.step()

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
                model.train(False)
                img, label, img_size = data
                label = box_transform(xywh_to_x1y1x2y2(label), img_size)

                _input = Variable(img.to(device))  # use cuda
                _target = Variable(label.to(device))  # use cuda

                with torch.no_grad():
                    output = model(_input)

                loss = self.criterion(output, _target)
                accuracy = compute_acc(output.data.cpu(), _target.data.cpu(), img_size)

                val_accuracy += accuracy
                val_loss += loss

            val_loss = float(val_loss) / len(validation_loader)
            val_accuracy = float(val_accuracy) / len(validation_loader)

            print(
                "Epoch: {}/{}, Training Accuracy: {:3f}, Training Loss: {:3f}, Validation Accuracy: {:3f}, Validation Loss: {:3f}".format(
                    idx + 1, epochs, total_accuracy, total_loss, val_accuracy, val_loss))

            self.writer.add_scalar('training_loss', total_loss, train_epoch + 1)
            self.writer.add_scalar('training_accuracy', total_accuracy, train_epoch + 1)
            self.writer.add_scalar('validation_loss', val_loss, train_epoch + 1)
            self.writer.add_scalar('validation_accuracy', val_accuracy, train_epoch + 1)

            if val_accuracy < self.best_accuracy:
                self.best_epoch = train_epoch

                torch.save(model.state_dict(),
                           os.path.join(self.save_model, str(self.experiment) + '_model.pt'))

                print("=> Best Epoch: {}, Accuracy: {:3f}".format(self.best_epoch, val_accuracy))

        self.writer.close()


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
    parser.add_argument('-exp', '--experiment', help='Experiment number to save the models', default=1)
    parser.add_argument('-sm', '--save_model', help='Path for saving the best model during training',
                        default='./models')

    args = parser.parse_args()
    model = InitiateTraining(args)
    model.train()


if __name__ == '__main__':
    main()
