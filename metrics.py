# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/10/12

import logging as lg

import numpy as np
from sklearn.metrics import confusion_matrix

_logger = lg.getLogger("metrics")


class Metric:
    """Base class for metrics"""
    def __init__(self):
        self.running_total = 0
        self.call_count = 0

    def __call__(self, predictions, labels, ):
        """Calculate streaming result"""
        self.call_count += 1
        res = self.calculation(predictions, labels)
        self.running_total += res
        return self.running_total/self.call_count

    def calculation(self, predictions, labels):
        """Calculation implementation"""
        raise NotImplementedError

    def reset(self):
        """Reset Streaming Metrics"""
        self.running_total = 0
        self.call_count = 0


class Accuracy(Metric):
    def __str__(self):
        return "accuracy"

    def calculation(self, predictions, labels):
        preds_np = np.argmax(predictions.numpy(), axis=1) # Note: not sure if this axis is right
        x = np.sum(preds_np == labels.numpy()) / preds_np.size
        return x


def get_cmx(predictions, labels):
    """Get Confusion Matrix"""
    preds_np = np.argmax(predictions.numpy(), axis=1).flatten()
    labels_np = labels.numpy().flatten()
    return confusion_matrix(labels_np, preds_np)


class Miou(Metric): # Note may want to ignore back ground class
    """Calculates Mean Intersection Over Union"""

    def __str__(self):
        return "Miou"

    @staticmethod
    def get_miou(cmx):
        intersection = np.diag(cmx)
        ground_truth_set = cmx.sum(axis=1)
        predicted_set = cmx.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        iou = intersection / union.astype(np.float32)
        return np.mean(iou)

    def calculation(self, predictions, labels):
        cmx = get_cmx(predictions, labels)
        return self.get_miou(cmx)


class Dice(Metric):
    def __str__(self):
        return "Dice"

    def calculation(self, predictions, labels, smooth):
        intersection = np.sum(labels * predictions, axis=[1, 2, 3])
        union = np.sum(labels, axis=[1, 2, 3]) + np.sum(predictions, axis=[1, 2, 3])
        dice = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        return dice


class MetricManager:
    """Mangers all metrics during training"""
    def __init__(self, metrics, writer=None):
        """

        Args:
            metrics (list(Metrics): list of metrics
            writer (Summary):
        """
        self.metrics = metrics
        self.writer = writer

    def update(self, preds, labels, step):
        for m in self.metrics:
            self._update_metric(m, preds.detach().cpu(), labels.detach().cpu(), step)

    def _update_metric(self, metric, preds, labels, step):
        result = metric(preds, labels)
        if self.writer:
            self.writer.add_scalar(str(metric), result, step)
        # _logger.DEBUG(f'{str(metric): {result}}')

    def reset(self):
        """Call reset method on all metrics"""
        _ = [m.reset() for m in self.metrics]
