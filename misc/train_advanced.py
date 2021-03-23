import os, sys
from numpy.lib.npyio import save
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import T

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


parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/misc/'.format(log_base_path)
log_loss_path =  '{}loss/misc/'.format(log_base_path)
log_weights_path =  '{}weights/misc/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/misc/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/misc/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_integral_pred = False

problem_path = 'Advanced1D'  # TODO
grid_dimensions = [100, 1]  # TODO

# hyper parameter of positional encoding in NeRF
epoch_sizes = 3000  # TODO
mrconfprint = 'grid Dimension: {}\n'.format(grid_dimensions)
sys.stderr.write(mrconfprint)

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

# deep learning modules
## fourfeat
scale = 0.0  # TODO
embedding_size = 512  # TODO
if scale == 0.0:
    embedding_size = 0
## siren
first_omega_0 = 0.05
hidden_omega_0 = 0.05

domain = np.array([[-10., 10.],[0., 1.]])
mlp_model = networks.MLP(in_features=1, out_features=1, n_neurons=1024, n_layers=10, scale=scale,
                        embedding_size=embedding_size, hidden_act=nn.SiLU(), output_act=None)
# siren_model = networks.Siren(in_features=1, out_features=1, hidden_features=1024, hidden_layers=10, normalized=False,
                            #  outermost_linear=True, first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)
model = mlp_model
siren = True
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))
if not siren:
    sys.stderr.write('Positional Encoding -> scale={}, embedding_size={}\n'.format(scale, embedding_size))
else:
    sys.stderr.write('Positional Encoding -> first omega zero={}, hidden omega zero={}\n'.format(first_omega_0, hidden_omega_0))

learning_rate = 1e-5
optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())))
criterion = nn.MSELoss(reduction='mean')
sys.stderr.write('DL optim: {}\n'.format(optim))

# record runtime
start_time = time.perf_counter()

# data
dataset = utils.MeshGrid(sidelen=grid_dimensions, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
x = next(iter(dataloader))[..., 0]

# examples from https://xaktly.com/ToughIntegrals.html
def f(x):
    # return 1. / (x**2 - x + 1 + 1e-5)  # problematic
    # return 1. / (torch.sqrt(x) * (x + 1.))  # problematic
    return 
def g(x):
    # return (3. * np.sqrt(3.) / 8.) * torch.atan(2. * (x - 0.5) / np.sqrt(3.))  # problematic
    # return 2. * torch.atan(torch.sqrt(x))  # problematic
sys.stderr.write('Target grad function: {}, target integral function: {}\n'.format('1. / (sqrt(x) * (x + 1.))',
                 '2 * atan(sqrt(x))'))
if torch.cuda.is_available():
    x = x.cuda().requires_grad_(True)

# training
batch_size = 1
loss_array = []

# training
for step in tqdm(range(epoch_sizes), desc='Training: '):
    model.train()

    optim.zero_grad()
    
    if siren:
        integral_pred, x = model(x)
    else:
        integral_pred= model(x)
        integral_pred = integral_pred.view(grid_dimensions)
    grad_pred = utils.gradient(integral_pred, x)
    
    loss = criterion(grad_pred, f(x))
    loss.backward()

    optim.step()

    # save loss values for plotting
    loss_array.append(loss.detach().item())
    sys.stderr.write("Total Steps: %d, MSE loss %0.6f\n" % (step, loss))

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# test model
if siren:
    integral_pred, x = model(x)
else:
    integral_pred = model(x)
grad_pred = utils.gradient(integral_pred, x)
sys.stderr.write('Grad network allclose: {}\n'.format(torch.allclose(input=grad_pred, other=f(x),
                                                                     atol=1e-4, rtol=1e-4)))  # check grad network
# integral_pred -= integral_pred.min()  # TODO: WHY IS THIS HAPPENING? ALWAYS IT IS SHIFTED (BIASES? -> disable biases)
## when we use 256x4 mlp with good activation function, this error starts to vanish.
sys.stderr.write('Integral network allclose: {}\n'.format(torch.allclose(input=integral_pred, other=g(x),
                                                                         atol=1e-4, rtol=1e-4)))  # check integral network


# visualization and saving model
grid_title = ''.join(str(i)+'x' for i in grid_dimensions)[:-1]
model_title = 'siren' if siren else 'fourfeat'
title = model_title+'_'+problem_path+'_'+str(step+1)+'_'+grid_title
title = visualizations.loss_vis(loss_array, title, True, path=log_loss_path)
visualizations.pred_vs_gt_integrad(pred=integral_pred, gt=g(x), x=x, grid_dimensions=grid_dimensions,
                                    loss=loss_array[-1], title='Integral_'+title, save=visualize, path=log_image_path)
visualizations.pred_vs_gt_integrad(pred=grad_pred, gt=f(x), x=x, grid_dimensions=grid_dimensions,
                                    loss=loss_array[-1], title='Grad_'+title, save=visualize, path=log_image_path)
# save weights of model
utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(integral_pred, grid_dimensions, title, save_integral_pred, True, path='logs/densities/fourfeat_multires/')
