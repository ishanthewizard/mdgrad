import datetime
import itertools
from collections import defaultdict
import os
from pathlib import Path

import json
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ase.neighborlist import natural_cutoffs, NeighborList
from ase.io import read, Trajectory
import sys
from torchmd.observable import *
from nff.train import get_model
import torchmd
from torchmd.system import GNNPotentials, PairPotentials, System, Stack
from torchmd.md import NoseHooverChain, Simulations
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units, Atoms
from lmdb_dataset import LmdbDataset, data_list_collater


# optional. nglview for visualization
# import nglview as nv
molecule = 'aspirin'
size = '1k'
name = 'md17'

def get_exp_rdf(data, nbins, r_range, obs):
    # load RDF data 
    f = interpolate.interp1d(data[:,0], data[:,1])
    start = r_range[0]
    end = r_range[1]
    xnew = np.linspace(start, end, nbins)

    # make sure the rdf data is normalized
    V = (4/3)* np.pi * (end ** 3 - start ** 3)
    g_obs = torch.Tensor(f(xnew)).to(obs.device)
    g_obs_norm = ((g_obs.detach() * obs.vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * obs.vol_bins / V

    return count_obs, g_obs

def JS_rdf(g_obs, g):
    e0 = 1e-4
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).sum()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).sum()

    return loss_js

def plot_rdfs(bins, target_g, simulated_g, fname, path):
    plt.title("epoch {}".format(fname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()

def get_hr(traj, bins):
    """
    compute h(r) the RDF??? for MD17 simulations.
    traj: T x N_atoms x 3
    """
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    return hist

def find_hr_from_file(molecule: str, size: str):
    #RDF plotting parameters
    xlim = 6
    n_bins = 500
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing h(r)

    # load ground truth data
    DATAPATH = f'md17/{molecule}/{size}/test/nequip_npz.npz'
    gt_data = np.load(DATAPATH)
    gt_traj = torch.FloatTensor(gt_data.f.R)
    gt_atomicnums = torch.FloatTensor(gt_data.f.z)
    hist_gt = get_hr(gt_traj, bins)
    return hist_gt
    # compute h(r) for simulated trajectory

def load_schnet_model(path = None, num_interactions = None, device = "cpu", mode="policy", from_pretrained=True):
    ckpt_and_config_path = os.path.join(path, "checkpoints", "best_checkpoint.pt")
    schnet_config = torch.load(ckpt_and_config_path, map_location=torch.device("cpu"))["config"]
    if num_interactions: #manual override
        schnet_config["model_attributes"]["num_interactions"] = num_interactions
    keep = list(schnet_config["model_attributes"].keys())[0:5]
    args = {k: schnet_config["model_attributes"][k] for k in keep}
    model = SchNet(**args).to(device)
    if from_pretrained:
        #get checkpoint
        ckpt_path = os.path.join(path, "checkpoints", "best_checkpoint.pt")
        checkpoint = {k: v.to(device) for k,v in torch.load(ckpt_path, map_location = torch.device("cpu"))['state_dict'].items()}
        #checkpoint =  torch.load(ckpt_path, map_location = device)["state_dict"]
        try:
            new_dict = {k[7:]: v for k, v in checkpoint.items()}
            model.load_state_dict(new_dict)
        except:
            model.load_state_dict(checkpoint)
    return model, schnet_config["model_attributes"]

def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers, 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms


#Get the global loss that you'll be training on
rdf_observable = find_hr_from_file("aspirin", "1k")

# 
def fit_rdf(assignments, i, suggestion_id, device, sys_params, project_name):
    n_epochs = 1000 
    n_sim = 200 
    size = 4
    cutoff = 2.5
    t_range = 50 # time range???

    nbins = 500 # bins for the rdf histogram
    tau = 120 # ??? what is this

    rdf_start = .75
    skip = 1

    # parse params 
    data = sys_params['data']
    size = sys_params['size']
    L = sys_params['L']
    end = sys_params['end']
    tmax = sys_params['tmax']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    cutoff = assignments['cutoff']

    nbins = assignments['nbins']
    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

    # initialize states with ASE # TODO: instead, load in your model DONE
    #initialize datasets
    train_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], name, molecule, size, 'train')})
    valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], name, molecule, size, 'val')})

    #get first configuration from dataset
    init_data = train_dataset.__getitem__(0)
    n_atoms = init_data['pos'].shape[0]
    atoms = data_to_atoms(init_data)
    system = System(atoms, device=device)
    system.set_temperature(298.0 * units.kB)
    print(system.get_temperature())

    # Initialize potentials 
    # TODO: replace with schnet DONE
    #load in schnet, train using simulate which calls odeintadjoint
    model, config = load_schnet_model(device="gpu")
    GNN = GNNPotentials(model, system.get_batch(), system.get_cell_len(), cutoff=cutoff, device=system.device)
    model = GNN

    # define the equation of motion to propagate (the Integrater)
    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=False).to(device)

    # define simulator with 
    sim = Simulations(system, diffeq)
    start = 2.0

    # initialize observable function 
    obs = rdf(system, nbins, (start, end) )

    xnew = np.linspace(start, end, nbins)
    # get experimental rdf TODO: replace 
    g_obs = find_hr_from_file("aspirin", "1k")
    # count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)

    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=assignments['lr'])

    loss_log = []
    loss_js_log = []
    traj = []

    solver_method = 'NH_verlet'

    for i in range(0, n_epochs):
        current_time = datetime.now() 
        trajs = sim.simulate(steps=tau, frequency=int(tau//2))
        v_t, q_t, pv_t = trajs 
        _, bins, g = obs(q_t)
        if i % 25 == 0:
           plot_rdfs(xnew, g_obs, g, i, model_path)
        # this shoud be wrapped in some way 
        loss_js = JS_rdf(g_obs, g)
        loss = (g- g_obs).pow(2).sum()
                
        print(loss_js.item(), loss.item())
        loss.backward()
        
        duration = (datetime.now() - current_time)
        
        optimizer.step()
        optimizer.zero_grad()

        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return np.array(loss_log[-16:-1]).mean()
        else:
            loss_log.append(loss_js.item())

        # check for loss convergence
        min_idx = np.array(loss_log).argmin()

        if i - min_idx >= 125:
            print("converged")
            break

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    train_traj = [var[1] for var in diffeq.traj]
    save_traj(system, train_traj, model_path + '/train.xyz', skip=10)

    # Inference 
    sim_trajs = []
    for i in range(n_sim):
        _, q_t, _ = sim.simulate(steps=100, frequency=25)
        sim_trajs.append(q_t[-1].detach().cpu().numpy())

    sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
    sim_trajs.requires_grad = False # no gradient required 

    # compute equilibrate rdf with finer bins 
    test_nbins = 128
    obs = rdf(system, test_nbins,  (start, end))
    xnew = np.linspace(start, end, test_nbins)
    count_obs, g_obs = get_exp_rdf(data, test_nbins, (start, end), obs) # recompute exp. rdf
    _, bins, g = obs(sim_trajs) # compute simulated rdf

    # compute equilibrated rdf 
    loss_js = JS_rdf(g_obs, g)

    save_traj(system, sim_trajs.detach().cpu().numpy(),  
        model_path + '/sim.xyz', skip=1)

    plot_rdfs(xnew, g_obs, g, "final", model_path)

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    if torch.isnan(loss_js):
        return np.array(loss_log[-16:-1]).mean()
    else:
        return loss_js.item()

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 
