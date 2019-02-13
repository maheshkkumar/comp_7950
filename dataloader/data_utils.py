import json

import torch

# seeding random variable to reproduce results
torch.manual_seed(0)


def read_file(path):
    data = None
    with open(path, 'r') as fp:
        if path.endswith('.txt'):
            data = [img.rstrip('\n') for img in fp.readlines()]
        elif path.endswith('.json'):
            data = json.load(fp)
        else:
            raise IOError
    return data


def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp


def xywh_to_x1y1x2y2(boxes):
    # input: box in (x,y,w,h) format
    # output: box in (x1,y1,x2,y2) format (representing two corners)
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes


def x1y1x2y2_to_xywh(boxes):
    # input: box in (x1,y1,x2,y2) format
    # output: box in (x,y,w,h) format
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes


def box_transform(boxes, im_sizes):
    # input: box in (x, y, w, h) format
    # output: normalize (x,y,w,h) in the range of [-1..1]
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = 2 * boxes[:, 0] / im_sizes[:, 0] - 1
    boxes[:, 1] = 2 * boxes[:, 1] / im_sizes[:, 1] - 1
    boxes[:, 2] = 2 * boxes[:, 2] / im_sizes[:, 0] - 1
    boxes[:, 3] = 2 * boxes[:, 3] / im_sizes[:, 1] - 1
    return boxes


def box_transform_inv(boxes, im_sizes):
    # input: box in (x,y,w,h) format normalized in the range of [-1..1]
    # output: convert to (x,y,w,h) in original image coordinates
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = (boxes[:, 0] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 1] = (boxes[:, 1] + 1) / 2 * im_sizes[:, 1]
    boxes[:, 2] = (boxes[:, 2] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 3] = (boxes[:, 3] + 1) / 2 * im_sizes[:, 1]
    return boxes


def compute_IoU(boxes1, boxes2):
    boxes1 = to_2d_tensor(boxes1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = to_2d_tensor(boxes2)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert ((a1 + a2 - ia <= 0).sum() == 0)

    return ia / (a1 + a2 - ia)

# def crop_boxes(boxes, im_sizes):
#     boxes = to_2d_tensor(boxes)
#     im_sizes = to_2d_tensor(im_sizes)
#     boxes = xywh_to_x1y1x2y2(boxes)
#     zero = torch.Tensor([0])
#     boxes[:, 0] = torch.max(torch.min(boxes[:, 0], im_sizes[:, 0]), zero)
#     boxes[:, 1] = torch.max(torch.min(boxes[:, 1], im_sizes[:, 1]), zero)
#     boxes[:, 2] = torch.max(torch.min(boxes[:, 2], im_sizes[:, 0]), zero)
#     boxes[:, 3] = torch.max(torch.min(boxes[:, 3], im_sizes[:, 1]), zero)
#     boxes = x1y1x2y2_to_xywh(boxes)
#     return boxes

def compute_acc(preds, targets, im_sizes, theta=0.4):
    preds = box_transform_inv(preds.clone(), im_sizes)
    targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    return IoU.sum() / preds.size(0)
