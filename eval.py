# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

""" Evaluation"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

import metrics
import model
import dataset


class Eval:
    def __init__(self, metric_man, loss, data, model, checkpoint_path):
        """Evaluation runner

        Args:
            metric_man (MetricManager):
            loss (torch.nn.modules.loss):
            data (DataLoader):
            model ():
        """

        self.metrics = metric_man
        self.loss = loss
        self.model = model
        self.data = data
        self.checkpoint_path = checkpoint_path

        self.step = 0

    def train_step(self, batch):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss(preds, labels)
        self.metrics.update(preds, labels, self.step, one_hot=False)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)

    def run(self):
        while True:
            self.load_new_checkpoint()
            for batch in self.data:
                self.train_step(batch)
                self.step += 1
            self.metrics.reset()

    def load_new_checkpoint(self):
        check_point = self.get_new_check()
        self.model.load_state_dict(torch.load(check_point))
        self.model.cpu()
        self.model.eval()

    def get_new_check(self):
        """ Check to see if new checkpoint then run eval"""
        return self.checkpoint_path


def run_eval(data, checkpoints_dir, log_dir):

    # Data
    images_path = data/"images"
    labels_path = data/"labels"
    ds = dataset.HeartDataSet(images_path, labels_path)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    # Metrics
    acc = metrics.Accuracy()
    miou = metrics.Miou()
    writer = SummaryWriter(log_dir/"validation")
    manager = metrics.MetricManager([acc, miou], writer)

    # Loss
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1e-3,1,1,1], device="cuda"))

    # Model
    unet = model.ResNetUNet(4)

    Eval(manager, criterion, loader, unet, checkpoints_dir).run()
