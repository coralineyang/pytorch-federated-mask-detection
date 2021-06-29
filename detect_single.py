from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.model_dump import *
from model_wrapper import Yolo

import cv2

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json
def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def detect_mask(original_img):
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4

    task_config = load_json('data/task_configs/yolo/clients_data/yolo_task1.json')
    model = Yolo(task_config)

    weights = pickle_string_to_obj("yolo_model.pkl")
    model.set_weights(weights)

    classes = load_classes("data/custom/classes.names")
    print(classes)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    transform_valid = transforms.Compose(
        [DEFAULT_TRANSFORMS,
         Resize(img_size)]
    )
    #img = cv2.imread(img_path)
    # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

    # Label Placeholder
    boxes = np.zeros((1, 5))

    img, _  = transform_valid((original_img, boxes))
    img = img.unsqueeze(0)
    img = img.to(device)
    img = Variable(img.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model.yolo(img)
        print(detections)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        print(detections)

    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    #
    # img = original_img
    # plt.figure()
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)

    results = []
    # Draw bounding boxes and labels of detections
    if detections is not None:
        for detection in detections:
            # Rescale boxes to original image
            #print(type(detection))
            detection = rescale_boxes(detection, img_size, original_img.shape[:2])
            #unique_labels = detection[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            print(detection)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                results.append({"name": classes[int(cls_pred)],
                                "conf": str(round(float(conf), 3)),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

                # box_w = x2 - x1
                # box_h = y2 - y1
                #
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # # Create a Rectangle patch
                # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # # Add the bbox to the plot
                # ax.add_patch(bbox)
                # # Add label
                # plt.text(
                #     x1,
                #     y1,
                #     s=classes[int(cls_pred)],
                #     color="white",
                #     verticalalignment="top",
                #     bbox={"color": color, "pad": 0},
                # )
    print(results)
    return {"results":results}

    # # Save generated image with detections
    # plt.axis("off")
    # plt.gca().xaxis.set_major_locator(NullLocator())
    # plt.gca().yaxis.set_major_locator(NullLocator())
    # # print(output_path)
    # plt.show()
    # plt.close()


# if __name__ == '__main__':
#     img1 = cv2.imread("data/samples_mask/sample3.jpeg")
#     img = Image.open("data/samples_mask/sample3.jpeg")
#     img = np.array(img)
#     #print(img)
#     #print(img1)
#     detect_mask(img)