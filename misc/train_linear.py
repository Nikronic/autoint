import os, sys
from numpy.lib.npyio import save
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networks
import utils
import multires_utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import time
import argparse
import itertools


# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = False
save_integral_pred = False

problem_path = 'Linear1D'  # TODO
grid_dimensions = [100, 1]  # TODO

# hyper parameter of positional encoding in NeRF
epoch_sizes = 500  # TODO
mrconfprint = 'grid Dimension: {}\n'.format(grid_dimensions)
sys.stderr.write(mrconfprint)

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

# deep learning modules
domain = np.array([[0., 1.],[0., 1.]])
mlp_model = nn.Linear(in_features=1, out_features=1, bias=False)  # learning ax=4x
model = mlp_model
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))

learning_rate = 3e-2
optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())))
criterion = nn.MSELoss(reduction='mean')
sys.stderr.write('DL optim: {}\n'.format(optim))

# record runtime
start_time = time.perf_counter()

# data
dataset = utils.MeshGrid(sidelen=grid_dimensions, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
x = next(iter(dataloader))[..., 0]
# x = torch.tensor([2.])
def f(x):  # for test case: f(x=2)=4
    return 4 * torch.ones_like(x)
def g(x):  # for test case: g(x=2)=8
    return 4 * x
if torch.cuda.is_available():
    x = x.cuda().requires_grad_(True)

# training
batch_size = 1
loss_array = []

# training
for step in tqdm(range(epoch_sizes), desc='Training: '):
    model.train()

    def closure():
        optim.zero_grad()

        integral_pred = model(x)
        integral_pred = integral_pred.view(grid_dimensions)
        grad_pred = utils.gradient(integral_pred, x)
        
        loss = criterion(grad_pred, f(x))
        loss.backward()

        # save loss values for plotting
        loss_array.append(loss.detach().item())
        sys.stderr.write("Total Steps: %d, MSE loss %0.6f\n" % (step, loss))

        return loss

    optim.step(closure)

# test model
integral_pred = model(x)
sys.stderr.write('Integral network allclose: {}\n'.format(torch.allclose(input=integral_pred, other=g(x))))  # check integral network
sys.stderr.write('Grad network allclose: {}\n'.format(torch.allclose(input=utils.gradient(integral_pred, x), other=f(x))))  # check grad network

# visualization and saving model
grid_title = ''.join(str(i)+'x' for i in grid_dimensions)[:-1]
title = 'mlp_'+problem_path+'_'+str(step+1)+'_'+grid_title
title = visualizations.loss_vis(loss_array, title, True, path='tmp/')
visualizations.pred_vs_gt_integrad(pred=integral_pred, gt=g(x), x=x, grid_dimensions=grid_dimensions,
                                    loss=loss_array[-1], title='Integral_'+title, save=visualize, path='tmp/')
visualizations.pred_vs_gt_integrad(pred=utils.gradient(model(x), x), gt=f(x), x=x, grid_dimensions=grid_dimensions,
                                    loss=loss_array[-1], title='Grad_'+title, save=visualize, path='tmp/')

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# utils.save_densities(integral_pred, grid_dimensions, title, save_integral_pred, True, path='logs/densities/fourfeat_multires/')
