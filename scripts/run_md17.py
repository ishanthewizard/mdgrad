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
from ase.neighborlist import natural_cutoffs, NeighborList
from lmdb_dataset import LmdbDataset
import gsd.hoomd
# optional. nglview for visualization
# import nglview as nv
MOLECULE = 'aspirin'
SIZE = '1k'
NAME = 'md17'
SCHNET_PATH = 'md17-aspirin_1k_schnet'


def plot_rdfs(bins, target_g, simulated_g, fname, path, tau, rdf_title, loss=-1):
    plt.title(f'epoch {fname}; loss: {loss}')
    plt.plot(bins, simulated_g.detach().cpu().numpy(), c='red', label='sim.' )
    plt.plot(bins, target_g, linewidth=2,linestyle='--', c='black', label='exp.')
    plt.legend()
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    # plt.ylim(0 , .6)
    plt.savefig(path + rdf_title, bbox_inches='tight')
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

def find_hr_from_file(molecule: str, size: str, n_bins, start, end):
    #RDF plotting parameters
    bins = np.linspace(start, end, n_bins + 1) # for computing h(r)

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
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # initialize states with ASE # TODO: instead, load in your model DONE
    #initialize datasets


    train_dataset = LmdbDataset({'src': os.path.join(NAME, MOLECULE, SIZE, 'train')})
    # valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], NAME, MOLECULE, SIZE, 'val')})

    #get first configuration from dataset
    np.random.randint(100)
    
    num_replicas = 99

    samples = np.random.choice(np.arange(train_dataset.__len__()), num_replicas)
    init_data_arr = [train_dataset.__getitem__(i) for i in samples]


    n_epochs = 1000  # number of epochs to train for
    cutoff = 15 # cutoff for interatomic distances (I don't think this is used)
    nbins = 500 # bins for the rdf histogram
    tau = 100 # this is the number of timesteps, idk why it's called tau
    start = 1e-6 # start of rdf range
    end = 10 # end of rdf range
    lr_initial = .001 # learning rate passed to optim
    dt = 0.5 * units.fs
    temp = 500* units.kB
    ttime =  20   #ttime is only used for NVT setup - it's not the total time
    n_atoms =init_data_arr[0]['pos'].shape[0] 
    targeEkin = 0.5 * (3.0 * n_atoms) * temp
    Q = 3.0 * n_atoms * temp * (ttime * dt)**2
    use_chain = False
    rdf_skip = 10
    checkpoint_epoch = 10
    # rdf_title = f'/replicas_ch{checkpoint_epoch}/rdf_plot_{num_replicas}replicas_{tau}t'
    rdf_title = 'tester'


    atoms_arr = [data_to_atoms(init_data) for init_data in init_data_arr]
    systems_arr= [System(atoms, device=device) for atoms in atoms_arr]
    [system.set_temperature(temp) for system in systems_arr]
    # NL_arr = [NeighborList(natural_cutoffs(atoms), self_interaction=False) for atoms in atoms_arr]
    # [NL_arr[i].update(atoms_arr[i]) for i in range(len(atoms_arr))]
    # bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T

    # PURE OVITO STUFF 
    # atom_types_list = list(set(atoms_arr[0].get_chemical_symbols()))
    # atom_types = atoms_arr[0].get_chemical_symbols()
    # type_to_index = {value: index for index, value in enumerate(atom_types_list)}
    # typeid = np.zeros(n_atoms, dtype=int)
    # for i, _type in enumerate(atom_types):
    #     typeid[i] = type_to_index[_type]   
    # END OVITO STUFF  
    try:
        device2 = torch.device(torch.cuda.current_device())
    except:
        device2 = "cpu"

    
    model, config = load_schnet_model(path= SCHNET_PATH, ckpt_epoch=checkpoint_epoch, device=torch.device(device2))
 

    atomic_nums = torch.Tensor(atoms_arr[0].get_atomic_numbers()).to(torch.long).to(device2).repeat(num_replicas)
    batch = torch.arange(num_replicas).repeat_interleave(n_atoms).to(device2)
    GNN = GNNPotentials(systems_arr[0], model, cutoff, atomic_nums, batch )
    model = GNN
    ovito_config = gsd.hoomd.open(name= f'{model_path}/sim_temp.gsd', mode='w')


    # define the equation of motion to propagate (the Integrater)

    diffeq = NoseHoover(model, 
            systems_arr,
            Q= Q, 
            T= temp,  
            targetEkin= targeEkin,
            num_replicas=num_replicas,
            adjoint=True).to(device)
    sim = Simulations(systems_arr, diffeq, method="MDsimNH", wrap=False)
    if use_chain:
        diffeq = NoseHooverChain(model, 
            systems_arr[0],
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=True).to(device)
        # define simulator with 
        sim = Simulations(systems_arr, diffeq, method="NH_verlet")




    # initialize observable function 
    obs = rdf(systems_arr[0], nbins, (start, end), width=.01 ) # initialize rdf function for the system





    xnew = np.linspace(start, end, nbins) # probably just the rdf bins
    # get experimental rdf TODO: replace 
    g_obs = find_hr_from_file("aspirin", "1k", n_bins=nbins, start=start, end=end)
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
        current_time = datetime.now() 
        
        trajs = sim.simulate(steps=tau, frequency=int(tau), dt = dt)
        # download_ovito(trajs, dt, bonds, atom_types_list, typeid, ovito_config)
        # ovito_config.close()
        v_t, q_t, pv_t = trajs
        # _, bins, g = obs(q_t[::rdf_skip, i])
        del(v_t)
        del(pv_t)
        q_t_obs = torch.stack([obs(q_t[::rdf_skip, i, : , :])[2] for i in range(q_t.shape[1])])
        #TODO: replace with vmap???

        # Now, let's average over all the rows for g (in the 99 dimension)
        g = q_t_obs.mean(dim=0)

        g = g * .5
        loss = (g - g_obs_tensor).pow(2).mean()
        print("LOSS: ", loss.item())
        if  i % 25 == 0:
           plot_rdfs(xnew, g_obs, g, i, model_path, tau, rdf_title, loss=loss.item())

        # Calculate the loss
        
    
        loss.backward()
        duration = (datetime.now() - current_time)
        
        optimizer.step()
        optimizer.zero_grad()
        
        
        if torch.isnan(loss):
            plt.plot(loss_log, list(range(i)))
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return np.array(loss_log[-16:-1]).mean()
        else:
            loss_log.append(loss.item())
            plt.plot(range(len(loss_log)), loss_log, )
            plt.savefig(model_path + '/loss.jpg')
            plt.close()

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

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    write(fname, atoms_list) 


def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    return tensor.numpy()

def download_ovito(trajs, dt, bonds, atom_types_list, typeid, ovito_config):
        tau = trajs[0].shape[0]
        for i in range(tau):
            radii = trajs[1][i]
            velocities = trajs[0][i]
            n_atoms = trajs[0].shape[1]
            # Particle positions, velocities, diameter
            partpos = detach_numpy(radii).tolist()
            velocities = detach_numpy(velocities).tolist()
            diameter = (10*0.08*np.ones((n_atoms,))).tolist()
            # Now make gsd file
            s = gsd.hoomd.Frame()
            s.configuration.step = i
            s.particles.N= n_atoms
            s.particles.position = partpos
            s.particles.velocity = velocities
            s.particles.diameter = diameter
            s.configuration.box=[10.0, 10.0, 10.0,0,0,0]
            #extract bond and atom type information
            s.bonds.N =  bonds.shape[0]
            s.bonds.types = atom_types_list
            s.bonds.typeid = typeid
            s.bonds.group = detach_numpy(bonds)
            ovito_config.append(s)


fit_rdf("123", "cuda", "test_proj")
