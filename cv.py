import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torchvision
from torchvision import transforms, datasets
from torchsummary import summary
import visdom

vis = visdom.Visdom("localhost")


# Load Dataset
seconds = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

train_movie, _, _ = torchvision.io.read_video(
    "data/train_movie.mp4", end_pts=seconds, pts_unit="sec")

train_movie = train_movie.permute(0, 3, 1, 2)

train_game, _, _ = torchvision.io.read_video(
    "data/train_game.mp4", end_pts=seconds, pts_unit="sec")

train_game = train_game.permute(0, 3, 1, 2)

test_game, _, _ = torchvision.io.read_video(
    "data/test_game.mp4", end_pts=seconds, pts_unit="sec")

test_game = test_game.permute(0, 3, 1, 2)

segmentation = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=True).eval()
segmentation.to(device)


def segmenter(image):
    prob_threshold = 0.5
    score_theshold = 0.8
    output = segmentation(image.unsqueeze(0))[0]
    valid_segments = torch.logical_and(
        output['labels'] == 1, output['scores'] >= score_theshold)
    probably_people_masks = (
        output['masks'][valid_segments] > prob_threshold)
    probably_people_boxes = (output['boxes'][valid_segments])

    for i in range(len(probably_people_boxes)):
        bounding_box = [int(a) for a in probably_people_boxes[i]]
        mask = probably_people_masks[i]
        inv_mask = mask.logical_not()
        masked_image = torch.stack([
            image[0].masked_fill(inv_mask, 0),
            image[1].masked_fill(inv_mask, 0),
            image[2].masked_fill(inv_mask, 0)])
        masked_image = masked_image.squeeze(1)

        yield masked_image[:, bounding_box[1]:bounding_box[3],
                           bounding_box[0]:bounding_box[2]].cpu()
            new_output = masked_image[:, bounding_box[1]:bounding_box[3],
                                      bounding_box[0]:bounding_box[2]].cpu()
            outputs.append(new_output)
    return outputs
