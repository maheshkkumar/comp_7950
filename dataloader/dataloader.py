import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from data_utils import read_file

# seeding random variable to reproduce results
np.random.seed(0)


class CUBDataset(Dataset):
    """
    Overriding Dataset class to handle custom dataset
    """

    def __init__(self, image_path, mode='train', split_rate=None, transform=None, label_path=None):
        self.image_list = read_file(image_path)

        self.mode = mode
        if not self.mode == 'test':
            self.label_list = read_file(label_path)
            assert (len(self.image_list) == len(self.label_list)), "Invalid image and label length"

        self.split_rate = split_rate
        self.transform = transform
        self.total_length = len(self.image_list)
        self.images = self.image_list

        if self.split_rate is not None:
            self.train_length = int(self.total_length * self.split_rate)

            if mode == 'train':
                self.images = self.image_list[:self.train_length]
                self.labels = self.label_list[:self.train_length]
            elif mode == 'validation':
                self.images = self.image_list[self.train_length:]
                self.labels = self.label_list[self.train_length:]
            else:
                pass

    def custom_transform(self, img):
        img = Image.open(img).convert('RGB')
        img_size = np.array(img.size, dtype='float32')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_size

    def custom_label(self, label, img_size):
        labels = np.array(list(map(lambda _: int(float(_)), label.split(' '))), dtype=np.float32)
        # labels = box_transform(xywh_to_x1y1x2y2(labels), img_size)
        return labels

    def __getitem__(self, index):
        image = self.images[index]
        image_path = os.path.join('./images/', image)
        img, img_size = self.custom_transform(image_path)

        if self.mode == 'test':
            return img, img_size
        else:
            labels = self.custom_label(self.labels[index], img_size)
            return img, labels, img_size

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_set = CUBDataset('./dataset/train_images.txt',
                           './dataset/train_boxes.txt', mode='train',
                           split_rate=0.8,
                           transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])]))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    for idx, d in enumerate(train_loader):
        print(d[0].size(), d[1].size(), d[1], d[2].size())
        break
    print("Dataset length: {}".format(len(train_loader)))
