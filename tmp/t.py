import os, sys
import torch
import kornia
from kornia.geometry import transform
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networks
import utils
import multires_utils
import visualizations

import numpy as np
import cv2
from skimage.transform import radon, warp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import argparse
import itertools

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_integral_pred = False

problem_path = 'CT'  # TODO
grid_dimensions = [100, 1]  # TODO

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

# deep learning modules
domain = np.array([[-1., 1.],[0., 180.]])


# image = torch.as_tensor(cv2.imread('data/phantom.png', 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).requires_grad_(True)
image = torch.as_tensor([[0, 1, 0.5],
                         [1, 1, 0],
                         [1, 0.5, 1]]).unsqueeze(0).unsqueeze(0).float()

# parameterization
rho = torch.linspace(domain[0][0], domain[0,1], 100)
alpha = torch.linspace(domain[1][0], domain[1,1], np.max(image.shape))
# t = torch.linspace()

# MAKE SURE `image` is 4D (B, C, H, W)
img_shape = np.array(image.shape[2:])
shape_min = min(img_shape)
radius = shape_min // 2
coords = (torch.arange(0, img_shape[0]).view(-1, 1), torch.arange(0, img_shape[1]).view(1, -1))
center = image.shape[-2] // 2

#############################################
R_batch = torch.zeros((len(alpha), 3, 3))
for i, angle in enumerate(torch.deg2rad(alpha)):
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                    [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                    [0., 0., 1.]]).unsqueeze(0)
    R_batch[i] = R
rotated = transform.warp_perspective(image.repeat((len(alpha), 1, 1, 1)), R_batch, dsize=(image.shape[-2], len(alpha)),
                                        mode='nearest', align_corners=True)

radon_image_ef = torch.rot90(rotated.sum(-2).squeeze(1))
#############################################

radon2 = radon(image.clone().detach().squeeze(0).squeeze(0).numpy(), theta=alpha.numpy())
radon2 = np.fliplr(radon2)

crit = nn.MSELoss(reduction='sum')
loss = crit(torch.as_tensor(radon2.copy()).float()/radon2.max(), radon_image_ef/radon_image_ef.max())
print()

