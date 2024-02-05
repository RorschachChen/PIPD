import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms=None, num_classes=10, shift=False, random_labels=False,
                 fixed_label=None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.img_set = None
        self.img_set = torch.load(data_dir)
        self.gt = torch.load(label_path)
        self.transforms = transforms
        self.transforms = []
        for t in transforms.transforms:
            if not isinstance(t, torchvision.transforms.ToTensor):
                self.transforms.append(t)
        self.transforms = torchvision.transforms.Compose(self.transforms)

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label

        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)

        if self.img_set is not None:  # if new version
            img = self.img_set[idx]
        else:  # if old version
            img = Image.open(os.path.join(self.dir, '%d.png' % idx))

        if self.transforms is not None:
            img = self.transforms(img)

        if self.random_labels:
            label = torch.randint(self.num_classes, (1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label + 1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label
