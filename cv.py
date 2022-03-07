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
