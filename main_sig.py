import argparse
import os

import numpy as np
import torch
from torchvision import transforms

from models import resnet
from none import my_compromised_detection
from toolbox_utils import tools
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 and 100 Training')
parser.add_argument('--round_num', type=int, default=5)
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--num_for_detect_biased', type=float, default=0.01)

args = parser.parse_args()

data_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])
path = 'poisoned_data/SIG_0.050_poison_seed=0'
poisoned_set_img_dir = f'{path}/imgs'
poisoned_set_label_path = f'{path}/labels'
poisoned_trainset = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                      label_path=poisoned_set_label_path, transforms=data_transform_aug)
trainloader_no_shuffle = torch.utils.data.DataLoader(poisoned_trainset, batch_size=2048, shuffle=False, num_workers=4)
model = resnet.resnet18(num_classes=10)
sd = torch.load(
    f'{path}/ckpt.pth')
model.load_state_dict(sd)
model.cuda()
poison_indices = torch.load(f'{path}/poison_indices')
poison_preset = np.load(f'{path}/isolation1%_examples.npy')
clean_preset = np.load(f'{path}/other20%_examples.npy')
sure_clean = None
for round_id in range(1, args.round_num + 1):
    poison_sample_index, poison_preset, clean_preset = my_compromised_detection.analyze_neuros(
        model, args.arch,
        10,
        args.num_for_detect_biased,
        trainloader_no_shuffle,
        sure_clean=sure_clean,
        last_poison_preset=poison_preset,
        last_clean_preset=clean_preset, conv=-1)

    final_poison_decision = np.setdiff1d(poison_sample_index, clean_preset)
    tp = np.intersect1d(poison_indices, poison_sample_index)
    print(f'tp: {len(tp)}, iso total: {len(poison_sample_index)}, all: {len(poison_indices)}')
    hk = np.arange(0, 50000)
    sure_clean = np.setdiff1d(hk, final_poison_decision)
