# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun Spinelli 2019/10/12

import logging as lg
_logger = lg.getLogger("train")

import torch

# from . import metrics


class Training:
    def __init__(self, metrics, loss, optim, data, epochs, model):
        """Training runner

        Args:
            metrics (MetricManager):
            loss (torch.nn.modules.loss):
            optim (torch.optim):
            data (DataLoader):
            epochs (int):
            model ():
        """

        self.metrics = metrics
        self.loss = loss
        self.optim = optim
        self.data = data
        self.model = model
        self.epochs = epochs

        self.step = 0

    def train_step(self, batch):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss(preds, labels)
        self.metrics.update(preds, labels, self.step)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)
        # _logger.debug(f'Loss: {loss.item()}')
        print(f'Loss: {loss.item()}')

        self.optim.zero_grad()  # zero gradients
        loss.backward()  # calculate gradients
        self.optim.step()  # updated weights

    def train(self):
        for i in range(self.epochs):
            _logger.info(f'Epoch {i}/{self.epochs}')
            print(f'Epoch {i}/{self.epochs}')
            for batch in self.data:
                self.train_step(batch)
                self.step += 1
            self.metrics.reset()
