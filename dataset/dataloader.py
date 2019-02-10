import os

import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from scaffolding.utils import read_file, box_transform, xywh_to_x1y1x2y2


class CUBDataset(Dataset):
    def __init__(self, image_path, label_path, mode='train', split_rate=None, transform=None):
        self.image_list = read_file(image_path)
        self.label_list = read_file(label_path)
        self.split_rate = split_rate
        self.transform = transform
        self.total_length = len(self.image_list)

        assert (len(self.image_list) == len(self.label_list)), "Invalid image and label length"

        if self.split_rate is not None:
            self.train_length = int(self.total_length * self.split_rate)

            if mode == 'train':
                self.images = self.image_list[:self.train_length]
                self.labels = self.label_list[:self.train_length]
            elif mode == 'validation':
                self.images = self.image_list[self.train_length:]
                self.labels = self.label_list[self.train_length:]
            else:
                self.images = self.image_list
                self.labels = self.label_list

    def custom_transform(self, img, label):
        if self.transform is not None:
            image = Image.open(img)
            img = self.transform(image)
            labels = [int(float(i)) for i in label.split(' ')]
            labels = box_transform(xywh_to_x1y1x2y2(labels), image.size)
        return img, labels, image.size

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image_path = os.path.join('/home/kumarkm/courses/DL/images/', image)
        img, label, img_size = self.custom_transform(image_path, label)
        return (img, label, img_size)

        # load image and convert to tensor and apply transformation
        # load labels, convert boxes to xyxy and normalize the values

        return image, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    train_set = CUBDataset('/home/kumarkm/workspace/first_year/comp_7950/dataset/train_images.txt',
                           '/home/kumarkm/workspace/first_year/comp_7950/dataset/train_boxes.txt', mode='train',
                           split_rate=0.8,
                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    for idx, data in enumerate(train_loader):
        #     print("{}\t{}\n".format(data[0], data[1]))
        image = data[0][0]
        print("Image size: {}".format(data[2][0]))
        labels = data[1][0]
        print("Labels: {}".format(labels))
        #     labels = xywh_to_x1y1x2y2(labels).numpy()[0]

        if (idx + 1) % 10 == 0:
            break
        image = image.numpy()
        print("Size of the image: {}".format(image.shape))
        image = np.rollaxis(image, -1, 0)
        image = np.rollaxis(image, -1, 0)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        #     rect = patches.Rectangle((labels[0], labels[1]), labels[2], labels[3],linewidth=1, edgecolor='r',facecolor='none')
        #     ax.add_patch(rect)
        plt.show()