import torch
from kornia.geometry import transform
from torch.utils.data import Dataset

import numpy as np

import os, sys
from datetime import datetime


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_mgrid(sidelen, domain, flatten=True):
    '''
    Generates a grid of nodes of elements in given ``domain`` range with ``sidelen`` nodes of that dim

    :param sidelen:  a 2D/3D tuple of number of nodes
    :param domain: a tuple of list of ranges of each dim corresponding to sidelen
    :param flatten: whether or not flatten the final grid (-1, 2/3)
    :return:
    '''

    sidelen = np.array(sidelen)
    tensors = []
    for d in range(len(sidelen)):
        tensors.append(torch.linspace(domain[d, 0], domain[d, 1], steps=sidelen[d]))
    tensors = tuple(tensors)
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if flatten:
        mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid


class MeshGrid(Dataset):
    def __init__(self, sidelen, domain, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return get_mgrid(self.sidelen, self.domain, self.flatten)


class SupervisedMeshGrid(Dataset):
    def __init__(self, sidelen, domain, gt_path, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates for a ground truth target with same grid size

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param gt_path: Path to the .npy saved ground truth densities of the same shape
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten
        self.gt_path = gt_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        
        # get saved ground truth
        gt_densities = np.load(self.gt_path).astype(np.float32)
        gt_densities = torch.as_tensor(gt_densities)
        gt_densities = gt_densities.permute(1, 0).unsqueeze(0)

        return get_mgrid(self.sidelen, self.domain, self.flatten), -gt_densities


def radon_transform(tensor, alpha=None, rotate=False):
    # MAKE SURE `image` is 4D (B, C, H, W)
    device = tensor.device
    if len(tensor.shape) != 4:
        raise ValueError('Please use a 4D tensor (B, C, H, W), your input shape is {}'.format(tensor.shape))
    center = tensor.shape[-2] // 2

    R_batch = torch.zeros((len(alpha), 3, 3), device=device)
    for i, angle in enumerate(torch.deg2rad(alpha)):
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                         [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                         [0., 0., 1.]], device=device).unsqueeze(0)
        R_batch[i] = R
    rotated = transform.warp_perspective(tensor.repeat((len(alpha), 1, 1, 1)), R_batch, dsize=(tensor.shape[-2], len(alpha)),
                                         mode='nearest', align_corners=True)

    if rotate:
        # for visualization part, use ``torch.rot90(radon_image_ef)``
        radon_image_ef = torch.rot90(rotated.sum(-2).squeeze(1))
    else:
        radon_image_ef = rotated.sum(-2).squeeze(1)
    return radon_image_ef


def count_parameters(model, trainable=True):
    """
    Counts the number of trainable parameters in a model

    :param model: Model to be processes
    :param trainable: Wether to only count trainable parameters
    """
    
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


# see issue #20: register_buffer is bugged in pytorch!
def save_weights(model, title, save=False, path=None):
    if path is None:
        path = 'tmp/'
    
    if save:
        d = {
            'scale': model.scale,
            'B': model.B,
            'model_state_dict': model.state_dict()
        }
        torch.save(d, path + title + '.pt')


def load_weights(model, path):
    d = torch.load(path)
    model.load_state_dict(d['model_state_dict'])
    model.B = d['B']
    model.scale = d['scale']
    sys.stderr.write('Weights, scale, and B  loaded.')
    

def save_densities(density, gridDimensions, title, save=False, prediciton=True, path=None):
    if path is None:
        path = 'tmp/'

    if save:
        if prediciton:
            if os.path.isfile(path + title + '_pred.npy'):
                title += str(int(datetime.timestamp(datetime.now())))
            with open(path + title + '_pred.npy', 'wb') as f:
                np.save(f, -density.view(gridDimensions).detach().cpu().numpy()[:, :].T)

        else:
            with open(path + title + '_gt.npy', 'wb') as f:
                np.save(f, -density.reshape(gridDimensions[0], gridDimensions[1]).T)

