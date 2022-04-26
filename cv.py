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
from enum import Enum, auto

vis = visdom.Visdom("ncc1.clients.dur.ac.uk", port=42069)

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
keypoint = torchvision.models.detection.keypointrcnn_resnet50_fpn(
    pretrained=True).eval()
keypoint.to(device)


def process_frame(frame):
    return (frame['data']/255).to(device)


# Section 1.1
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


class Pose(Enum):
    FULL_BODY_STANDING = auto(),
    FULL_BODY_SITTING = auto(),
    HALF_BODY = auto(),
    HEAD_ONLY = auto(),
    OTHER = auto()


# Section 1.2
def get_pose_and_quality(patches):
    for patch in patches:
        image, mask, box = patch

        output = keypoint(image.unsqueeze(0))[0]

        # Get only people
        output_iter = list(filter(lambda a: a[0] == 1, zip(
            output["labels"], output["scores"], output["keypoints"], output["keypoints_scores"])))

        if len(output_iter) == 0:
            continue

        # Get only the highest scoring person
        highest_scoring = max(output_iter, key=lambda a: a[1])

        # Get score for each keypoint
        kp = list(zip(highest_scoring[2], highest_scoring[3]))

        # Label keypoints
        keys = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"]
        kp = dict(zip(keys, kp))

        def most_are_detected(kp, parts):
            score = 0
            for part in parts:
                score += kp[part][1]
            if score / len(parts) > 2:
                return True
            else:
                return False

        # Head detection
        if most_are_detected(kp, ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]):
            head = True
        else:
            head = False

        # chest detection
        if most_are_detected(kp, ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow"]):
            chest = True
        else:
            chest = False

        # legs and hands detection
        if most_are_detected(kp, ["left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]):
            legs_and_hands = True
        else:
            legs_and_hands = False

        if head and chest and legs_and_hands:
            # Pose = Pose.FULL_BODY_...?
            torso_top = (kp["left_shoulder"][0] + kp["right_shoulder"][0])/2
            torso_bottom = (kp["left_hip"][0] + kp["right_hip"][0])/2
            knees = (kp["left_knee"][0] + kp["right_knee"][0])/2

            torso_vec = torso_bottom - torso_top
            thigh_vec = knees - torso_bottom

            torso_length = torch.norm(torso_vec)
            thigh_length = torch.norm(thigh_vec)

            abs_cos_angle = abs(torch.dot(torso_vec, thigh_vec) /
                                (torso_length * thigh_length))
            ratio = torso_length / thigh_length

            if abs_cos_angle > 0.5 and ratio < 2:
                pose = Pose.FULL_BODY_STANDING
            else:
                pose = Pose.FULL_BODY_SITTING

            pose = Pose.FULL_BODY_STANDING
        elif head and not chest:
            pose = Pose.HEAD_ONLY
        elif head and chest:
            pose = Pose.HALF_BODY
        else:
            pose = Pose.OTHER
        yield image, mask, box, pose


# Section 1.3
def filter_based_on_quality(patches):
    return filter(lambda patch: patch[3] is not Pose.OTHER, patches)


def patch_normalise(patches):
    for image, mask, box, pose in patches:
        resizer = torchvision.transforms.Resize((500, 400))
        normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        yield resizer(normalizer(image)), mask, box, pose


training_data_movie = patch_normalise(filter_based_on_quality(
    get_pose_and_quality(get_patches(train_movie_reader))))
training_data_game = patch_normalise(filter_based_on_quality(
    get_pose_and_quality(get_patches(train_game_reader))))

# i = 1
# for image, mask, bounding_box, pose in training_data_game:
    # torchvision.utils.save_image(image, f"data_game_img/{pose.name}-{i}.png")
    # i += 1
# i = 1
# for image, mask, bounding_box, pose in training_data_movie:
    # torchvision.utils.save_image(image, f"data_img/{pose.name}-{i}.png")
    # i += 1


from torchvision import datasets
game_dataset = datasets.ImageFolder("data_game_img")
movie_dataset = datasets.ImageFolder("data_img")

game_dataloader = torch.utils.data.DataLoader(game_dataset, batch_size=32, shuffle=True)
movie_dataloader = torch.utils.data.DataLoader(movie_dataset, batch_size=32, shuffle=True)

augmentations = torch.transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])


