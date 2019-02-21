import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from dataloader.data_utils import *


def visualize_results(results_path, dataset_path):
    """
    Custom method to visualize the results
    """
    results = [map(lambda _: _.tolist(), xywh_to_x1y1x2y2(map(float, res.split(" "))))[0] for res in
               read_file(results_path)]
    dataset_path = [os.path.join('./images/', i) for i in read_file(dataset_path)]

    w = 100  # changable parameters for grid size
    h = 100  # changable parameters for grid size
    fig = plt.figure(figsize=(30, 30))
    columns = 10
    rows = 12
    for i in range(1, len(dataset_path)):
        img = dataset_path[i]
        img = Image.open(img).convert('RGB')
        draw = ImageDraw.Draw(img)
        draw.rectangle(results[i], outline="red")
        # fig.add_subplot(rows, columns, i) # uncomment this to generate a grid of images
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    visualize_results('./output.txt', './dataset/test_images.txt')
