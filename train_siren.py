import os, sys
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


parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = False

problem_path = '1DIntegral'  # TODO
grid_dimensions = [10, 1]  # TODO

use_scheduler = False

# hyper parameter of positional encoding in NeRF
first_omega_0, hidden_omega_0 = 30., 30.  # TODO

# iterations
epoch_sizes = 5000  # TODO
mrconfprint = 'grid Dimension: {}\n'.format(grid_dimensions)
sys.stderr.write(mrconfprint)

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/siren/'.format(log_base_path)
log_loss_path =  '{}loss/siren/'.format(log_base_path)
log_weights_path =  '{}weights/siren/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/siren/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/siren/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))
sys.stderr.write('first omega zero: {}, hidden omega zero: {}\n'.format(first_omega_0, hidden_omega_0))

# reproducibility
seed = 8
torch.manual_seed(seed)
np.random.seed(seed)

# deep learning modules
domain = np.array([[0., 1.],[0., 1.]])
siren_model = networks.Siren(in_features=1, hidden_features=256, hidden_layers=4, out_features=1,
                             outermost_linear=True, first_omega_0=30., hidden_omega_0=30.)
model = siren_model
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))

learning_rate = 3e-4
optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())))
criterion = nn.MSELoss(reduction='mean')
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))

# record runtime
start_time = time.perf_counter()

# data
dataset = utils.MeshGrid(sidelen=grid_dimensions, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()

# training
batch_size = 1
loss_array = []

# training
for step in tqdm(range(epoch_sizes), desc='Training: '):
    model.train()

    def closure():
        optim.zero_grad()

        # aka x
        density = model(model_input)
        density = density.view(grid_dimensions)
        
        loss = criterion()
        loss.backward()

        # reduce LR if no reach plateau
        if use_scheduler:
            scheduler.step(loss)

        # save loss values for plotting
        loss_array.append(loss.detach().item())
        sys.stderr.write("Total Steps: %d, MSE loss %0.6f" % (step, loss))

        return loss

    optim.step(closure)

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# test model
with torch.no_grad():
    density = model(model_input)

    # visualization
    title = 'SIREN_omega'+str(first_omega_0)+'x'+str(hidden_omega_0)+'_'+str(step)
    title = visualizations.loss_vis(loss_array, title, True, path=log_loss_path)
    # visualizations.density_vis(density, loss_array[-1], grid_dimensions, title, True, visualize, True,
    #                             binary_loss=None, path=log_image_path)
    
# saving model
utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(density, grid_dimensions, title, save_density, True, path='logs/densities/siren_multires/')
