from __future__ import division

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def set_weights(model, parameters):
    for i, param in enumerate(model.parameters()):
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        param_ = torch.from_numpy(parameters[i]).cuda()
        param.data.copy_(param_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    #model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    import json
    def load_json(filename):
        with open(filename) as f:
            return json.load(f)

    task_config = load_json('data/task_configs/yolo/clients_data/yolo_task1.json')
    model = Yolo(task_config)

    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    weights = pickle_string_to_obj("yolo_model.pkl")
    model.set_weights(weights)


    #model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)])),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    print(classes)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    # testset = ListDataset(task_config['test'], img_size=416,
    #                       multiscale=False, transform=DEFAULT_TRANSFORMS)
    # test_dataloader = DataLoader(
    #     testset,
    #     batch_size=task_config['batch_size'],
    #     num_workers=1,
    #     shuffle=False,
    #     collate_fn=testset.collate_fn
    # )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    total_losses = list()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # for batch_i, (img_paths, input_imgs, targets) in enumerate(tqdm.tqdm(test_dataloader, desc="Detecting objects")):
    #     # Extract labels
    #     labels += targets[:, 1].tolist()
    #     # Rescale target
    #     targets = Variable(targets.to(device), requires_grad=False)
    #
    #     input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)
    #     with torch.no_grad():
    #         loss, detections = model.yolo(input_imgs, targets)
    #
    #         detections = non_max_suppression(detections, conf_thres=0.8, nms_thres=0.4)
    #

    #model.yolo.eval()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        print(type(input_imgs),input_imgs)
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model.yolo(input_imgs)
            print(detections)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            print(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            print(type(detections),detections)
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1


                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join("output", f"{filename}.png")
        #print(output_path)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
