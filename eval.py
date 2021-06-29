from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.model_dump import *
from model_wrapper import Yolo

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json
def load_json(filename):
    with open(filename) as f:
        return json.load(f)

task_config = load_json('data/task_configs/yolo/street_5/yolo_task1.json')
model = Yolo(task_config)

weights = pickle_string_to_obj("yolo_model.pkl")
model.set_weights(weights)

_, acc, recall = model.evaluate()
print("map:",acc,"\n")