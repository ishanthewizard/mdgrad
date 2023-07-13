from datetime import datetime
import itertools
from collections import defaultdict
import os
from pathlib import Path


import json
import pdb;
import shutil
from torch_geometric.nn import SchNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, Trajectory, write
from torchmd.observable import *
from torchmd.system import  System
from interface import GNNPotentials
from torchmd.md import NoseHoover, NoseHooverChain, Simulations
from ase import units, Atoms
from lmdb_dataset import LmdbDataset

# optional. nglview for visualization
# import nglview as nv
MOLECULE = 'aspirin'
SIZE = '1k'
NAME = 'md17'
SCHNET_PATH = 'md17-aspirin_1k_schnet'


def plot_rdfs(bins, target_g, simulated_g, fname, path):
    plt.title("epoch {}".format(fname))
    plt.plot(bins, simulated_g.detach().cpu().numpy(), c='blue', label='sim.' )
    plt.plot(bins, target_g, linewidth=2,linestyle='--', c='black', label='exp.')
    plt.legend()
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.ylim(0 , .6)
    plt.savefig(path + '/rdf_plot.jpg', bbox_inches='tight')
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


def load_schnet_model(path = None, ckpt_epoch = -1, num_interactions = None, device = "cpu", from_pretrained=True):
    
    cname = 'best_checkpoint.pt' if ckpt_epoch == -1 else f"checkpoint{ckpt_epoch}.pt"
    ckpt_and_config_path = os.path.join(path, "checkpoints", cname)
    schnet_config = torch.load(ckpt_and_config_path, map_location=torch.device("cpu"))["config"]
    if num_interactions: #manual override
        schnet_config["model_attributes"]["num_interactions"] = num_interactions
    keep = list(schnet_config["model_attributes"].keys())[0:5]
    args = {k: schnet_config["model_attributes"][k] for k in keep}
    model = SchNet(**args).to(device)

    if from_pretrained:
        #get checkpoint
        print(f'Loading model weights from {ckpt_and_config_path}')
        checkpoint = {k: v.to(device) for k,v in torch.load(ckpt_and_config_path, map_location = torch.device("cpu"))['state_dict'].items()}
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




def fit_rdf(suggestion_id, device, project_name):
    model_path = '{}/{}'.format(project_name, suggestion_id)
    # Remove the directory if it already exists
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Create the new directory
    os.makedirs(model_path)

    # initialize states with ASE # TODO: instead, load in your model DONE
    #initialize datasets
    train_dataset = LmdbDataset({'src': os.path.join(NAME, MOLECULE, SIZE, 'train')})
    # valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], NAME, MOLECULE, SIZE, 'val')})

    #get first configuration from dataset
    init_data = train_dataset.__getitem__(0)


    n_epochs = 1000  # number of epochs to train for
    cutoff = 7 # cutoff for interatomic distances (I don't think this is used)
    nbins = 500 # bins for the rdf histogram
    tau = 300 # this is the number of timesteps, idk why it's called tau
    start = 0 # start of rdf range
    end = 6 # end of rdf range
    lr_initial = .0001 # learning rate passed to optim
    dt = 0.5 * units.fs
    temp = 500* units.kB
    ttime =  20   #ttime is only used for NVT setup - it's not the total time
    n_atoms =init_data['pos'].shape[0] 
    targeEkin = 0.5 * (3.0 * n_atoms) * temp
    Q = 3.0 * n_atoms * temp * (ttime * dt)**2


    atoms = data_to_atoms(init_data)
    system = System(atoms, device=device)
    system.set_temperature(298.0 * units.kB)
    print(system.get_temperature())
    

    try:
        device2 = torch.device(torch.cuda.current_device())
    except:
        device2 = "cpu"

    
    model, config = load_schnet_model(path= SCHNET_PATH, ckpt_epoch='600', device=torch.device(device2))
    batch = system.get_batch()
    cell_len = system.get_cell_len()


    atomic_nums = torch.Tensor(atoms.get_atomic_numbers()).to(torch.long).to(device2)
    GNN = GNNPotentials(system, model, cutoff, atomic_nums )
    model = GNN

    # define the equation of motion to propagate (the Integrater)
    diffeq = NoseHoover(model, 
            system,
            Q= Q, 
            T= temp,
            targetEkin= targeEkin,
            adjoint=True).to(device)

    # define simulator with 
    sim = Simulations(system, diffeq, method="MDsimNH")

    # initialize observable function 
    obs = rdf(system, nbins, (start, end) ) # initialize rdf function for the system

    xnew = np.linspace(start, end, nbins) # probably just the rdf bins
    # get experimental rdf TODO: replace 
    g_obs = find_hr_from_file("aspirin", "1k")
    # count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)

    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=lr_initial)

    loss_log = []
    loss_js_log = []
    traj = []

    # Convert `g_obs` to a PyTorch tensor and move it to the same device as `g`
    g_obs_tensor = torch.from_numpy(g_obs).to(device)
    
    print("Training for {} epochs".format(n_epochs))
    for i in range(0, n_epochs):
        if i == 100:
            pdb.set_trace()
        current_time = datetime.now() 
        
        trajs = sim.simulate(steps=tau, frequency=int(tau), dt = dt)
        v_t, q_t, pv_t = trajs 
        _, bins, g = obs(q_t)
        if  i % 25 == 0:
           plot_rdfs(xnew, g_obs, g, i, model_path)
        pdb.set_trace()
        # Calculate the loss
        loss = (g - g_obs_tensor).pow(2).sum()
        print("LOSS: ", loss.item())
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
            loss_log.append(loss.item())

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

#     # Inference 
#     sim_trajs = []
#     for i in range(n_sim):
#         _, q_t, _ = sim.simulate(steps=100, frequency=25)
#         sim_trajs.append(q_t[-1].detach().cpu().numpy())

#     sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
#     sim_trajs.requires_grad = False # no gradient required 

#     # compute equilibrate rdf with finer bins 
#     test_nbins = 128
#     obs = rdf(system, test_nbins,  (start, end))
#     xnew = np.linspace(start, end, test_nbins)
#     count_obs, g_obs = get_exp_rdf(data, test_nbins, (start, end), obs) # recompute exp. rdf
#     _, bins, g = obs(sim_trajs) # compute simulated rdf

#     # compute equilibrated rdf 
#     loss_js = JS_rdf(g_obs, g)

#     save_traj(system, sim_trajs.detach().cpu().numpy(),  
#         model_path + '/sim.xyz', skip=1)

#     plot_rdfs(xnew, g_obs, g, "final", model_path)

#     np.savetxt(model_path + '/loss.csv', np.array(loss_log))

#     if torch.isnan(loss_js):
#         return np.array(loss_log[-16:-1]).mean()
#     else:
#         return loss_js.item()

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    write(fname, atoms_list) 

fit_rdf("123", "cuda", "test_proj")