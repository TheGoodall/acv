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
import itertools

vis = visdom.Visdom("localhost")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

# Load Dataset
train_game_reader = torchvision.io.VideoReader("data/train_game.mp4", "video")
train_movie_reader = torchvision.io.VideoReader(
    "data/train_movie.mp4", "video")
test_game_reader = torchvision.io.VideoReader("data/test_game.mp4", "video")


segmentation = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=True).eval()
segmentation.to(device)


def process_frame(frame):
    return (frame['data']/255).to(device)


def segmenter(image):
    prob_threshold = 0.5
    score_theshold = 0.9
    output = segmentation(image.unsqueeze(0))[0]
    valid_segments = torch.logical_and(
        output['labels'] == 1, output['scores'] >= score_theshold)
    probably_people_masks = (
        output['masks'][valid_segments] > prob_threshold)
    probably_people_boxes = (output['boxes'][valid_segments])

    for i in range(len(probably_people_boxes)):
        bounding_box = [int(a) for a in probably_people_boxes[i]]

        mask = probably_people_masks[i].squeeze(0)
        bounded_mask = mask[bounding_box[1]
            :bounding_box[3], bounding_box[0]:bounding_box[2]]

        bounded_image = image[:, bounding_box[1]
            :bounding_box[3], bounding_box[0]:bounding_box[2]]

        yield bounded_image, bounded_mask, bounding_box


def get_patches(reader):
    return itertools.chain.from_iterable(map(lambda a: (segmenter(process_frame(a))), reader))


def segment_frames(reader):
    return map(lambda a: (process_frame(a), (segmenter(process_frame(a)))), reader)
