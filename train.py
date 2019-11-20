# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun Spinelli 2019/10/12

import logging as lg

import torch

# from . import metrics
_logger = lg.getLogger("train")


class Training:
    def __init__(self, metrics, loss, optim, data, epochs, model, save_dir):
        """Training runner

        Args:
            metrics (MetricManager):
            loss (torch.nn.modules.loss):
            optim (torch.optim):
            data (DataLoader):
            epochs (int):
            model ():
            save_dir (str): directory to save model
        """

        self.metrics = metrics
        self.loss = loss
        self.optim = optim
        self.data = data
        self.model = model
        self.epochs = epochs
        self.save_dir = save_dir

        self.step = 0

    def train_step(self, batch):
        # data.cuda(), labels.cuda() = batch
        data, labels = batch
        preds = self.model(data)
        loss = self.loss(preds, labels)
        self.metrics.update(preds, labels, self.step, one_hot=False)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)
        # _logger.debug(f'Loss: {loss.item()}')

        self.optim.zero_grad()  # zero gradients
        loss.backward()  # calculate gradients
        self.optim.step()  # updated weights

    def save_checkpoint(self):
        """Save checkpoint with current step number"""
        torch.save(self.model.state_dict(), f'{self.save_dir}/model-{self.step}')

    def train_loop(self):
        for i in range(self.epochs):
            # _logger.info(f'Epoch {i}/{self.epochs}')
            print(f'Epoch {i}/{self.epochs}')
            for batch in self.data:
                self.train_step(batch)
                self.step += 1
            self.metrics.reset()
            self.save_checkpoint()

    def train_cancel(self):
        try:
            self.train_loop()
        except KeyboardInterrupt:
            _logger.debug("Quitting due to user cancel")
            self.save_checkpoint()

