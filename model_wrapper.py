import json
import numpy
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
#from model.yolo import Darknet
from utils.utils import *

from models import *
#from utils.logger import *
from utils.transforms import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *

from utils.parse_config import *


sys.path.append("")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
torch.set_num_threads(4)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(1234)


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class Yolo(object):
    def __init__(self, task_config):
        self.task_config = task_config  # task_config：yolo_task1.json
        self.model_config = load_json(task_config['model_config'])
        print(self.model_config)

        if 'train' in self.task_config:
            #print(self.task_config['train'])
            self.dataset = ListDataset(self.task_config['train'],
                                       multiscale=self.model_config['multiscale_training'],
                                       img_size = self.model_config['img_size'], transform = AUGMENTATION_TRANSFORMS)
            logging.info('load data')
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.task_config['batch_size'],
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=self.task_config['n_cpu'],
                                         collate_fn=self.dataset.collate_fn)
            # TODO: add a valset for validate
            self.testset = ListDataset(self.task_config['test'],img_size=self.model_config['img_size'],
                                       multiscale=False, transform=DEFAULT_TRANSFORMS)
            self.test_dataloader = DataLoader(
                self.testset,
                batch_size=self.task_config['batch_size'],
                num_workers=1,
                shuffle=False,
                collate_fn=self.testset.collate_fn
            )
            self.train_size = self.dataset.__len__()
            print("train_size:", self.train_size)
            self.valid_size = self.testset.__len__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = Darknet(self.model_config['model_def']).to(self.device)
        self.yolo.apply(weights_init_normal)

        assert os.path.exists(self.model_config['pretrained_weights'])
        self.yolo.load_darknet_weights(self.model_config['pretrained_weights'])
        logging.info('model construct completed')
        self.best_map = 0
        self.optimizer = torch.optim.Adam(self.yolo.parameters())

    def get_weights(self):
        params = [param.data.cpu().numpy()
                  for param in self.yolo.parameters()]
        return params

    def set_weights(self, parameters):
        for i, param in enumerate(self.yolo.parameters()):
            print(torch.cuda.device_count())
            print(torch.cuda.is_available())
            #param_ = torch.from_numpy(parameters[i]).cuda()
            param_ = torch.from_numpy(parameters[i])
            param.data.copy_(param_)

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        self.yolo.train()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.dataloader)):
            #print(imgs.shape)
            batches_done = len(self.dataloader) * 1 + batch_i
            imgs = Variable(imgs.to(self.device))
            targets = Variable(targets.to(self.device), requires_grad=False)
            loss, outputs = self.yolo(imgs, targets)
            loss.backward()
            if batch_i % 10 == 0:
                print("step: {} | loss: {:.4f}".format(batch_i, loss.item()))
            if batches_done % self.model_config["gradient_accumulations"]==0:
                # Accumulates gradient before each step
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss.item()

    def eval(self, dataloader, yolo, test_num=10000):
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        total_losses = list()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets = Variable(targets.to(self.device), requires_grad=False)

            imgs = Variable(imgs.type(Tensor), requires_grad=False)
            with torch.no_grad():
                loss, outputs = yolo(imgs, targets)
                #print(outputs)
                outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5)
                #print(outputs)
                total_losses.append(loss.item())
            targets = targets.to("cpu")
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= int(self.model_config['img_size'])
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
        if len(sample_metrics) > 0:
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        else:
            return 0.0, 0.0, 0.0
        total_loss = sum(total_losses) / len(total_losses)
        return total_loss, AP.mean(), recall.mean()

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set, 
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, mAP, recall = self.eval(
            self.test_dataloader, self.yolo, test_num)
        return total_loss, mAP, recall




class Models:
    Yolo = Yolo



from utils.model_dump import *
if __name__ == '__main__':
    def load_json(filename):
        with open(filename) as f:
            return json.load(f)

    task_config = load_json('data/task_configs/yolo/clients_data/yolo_task1.json')
    model = Yolo(task_config)

    weights = pickle_string_to_obj("yolo_model.pkl")
    model.set_weights(weights)
    #image_folder = "data/samples_mask"

    server_loss, server_map, server_recall = model.evaluate()
    print(server_loss, server_map, server_recall)

    #loss = model.train_one_epoch()