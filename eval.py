# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

""" Evaluation"""
from pathlib import Path
import time

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
            checkpoint_path (str): full path to checkpoint file if using `run once` or checkpoint
            dir if running during training.
        """

        self.metrics = metric_man
        self.loss = loss
        self.model = model
        self.data = data
        self.checkpoint_path = Path(checkpoint_path)

        self.step = 0
        self.current_check = -1

    def eval_step(self, batch):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss(preds, labels)
        self.metrics.update(preds, labels, self.step)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)

    def run_once(self):
        """Used to run after epoch or just once on entire eval set"""
        self.load_checkpoint(self.checkpoint_path)
        for batch in self.data:
            self.train_step(batch)
            self.step += 1
        self.metrics.reset()

    def run(self):
        """Run eval continually while model is training"""
        while True:
            self.wait_load_new_checkpoint()
            for batch in self.data:
                self.train_step(batch)
                self.step += 1
            self.metrics.reset()

    def load_checkpoint(self, path):
        """Load checkpoint from directory"""
        self.model.load_state_dict(torch.load(path))
        self.model.cpu()
        self.model.eval()

    def wait_load_new_checkpoint(self):
        """Waits till new checkpoint then loads checkpoint"""
        while True:
            models_idx = [int(m.stem[-1]) for m in self.checkpoint_path.iterdir()].sort()
            latest = models_idx[-1]
            if latest > self.current_check:
                latest = self.current_check
                self.load_checkpoint(f'{self.checkpoint_path}/model-{latest}.pth')
                break
            time.sleep(30)


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
