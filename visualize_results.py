import os
from dataloader.data_utils import *
import matplotlib.pyplot as plt
from PIL import Image

def visualize_results(results_path, dataset_path):

    # results = read_file(results_path)
    # results = map(lambda _: xywh_to_x1y1x2y2(map(lambda i: float(i)), _.split(" ")), read_file(results_path))

    results = [xywh_to_x1y1x2y2(map(float, res.split(" "))) for res in read_file(results_path)]
    dataset_path = [os.path.join('./images/', i) for i in read_file(dataset_path)]

    print(results[:10])
    print(dataset_path[:10])
    w=20
    h=20
    # fig=plt.figure(figsize=(8, 8))
    columns = 10
    rows = 12
    # for i in range(1, columns*rows +1):
    #     img = Image.open(images[i]).convert('RGB')
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    visualize_results('./results/results.txt', './dataset/test_images.txt')