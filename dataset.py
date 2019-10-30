# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun Spinelli 2019/01/01

import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

from PIL import Image


class HeartDataSet(Dataset):
    def __init__(self, images_path, labels_path, size=256):
        self.images_list = list(images_path.iterdir())
        self.labels_list = list(labels_path.iterdir())
        self.size = (size, size)

    def resize(self, arr):
        return np.array(Image.fromarray(arr).resize(self.size))  # bilinerar resizing

    def get_pair(self, image_path):
        label_path = [file for file in self.labels_list if str(image_path.name) in str(file)]
        assert len(label_path) == 1  # make sure there only one label per image
        label = torch.tensor(self.resize(np.load(label_path[0]))).long()#.to(torch.int64)

        image = torch.tensor((self.resize(np.load(image_path)) / 255)[None], dtype=torch.float)#.double()
        return image, label

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        image, label = self.get_pair(image_path)
        return image, label

