from datetime import datetime
import itertools
from collections import defaultdict
import os
from pathlib import Path


import json
import pdb;
import shutil
from torch_geometric.nn import SchNet
from torch.nn.utils import clip_grad_norm_
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
MOLECULE = 'naphthalene'
SIZE = '1k'
NAME = 'md17'
SCHNET_PATH = f'/data/ishan-amin/MODELPATH/schnet/md17-{MOLECULE}_{SIZE}_schnet'


def plot_rdfs(bins, target_g, simulated_g, fname, path, tau, rdf_title, loss=-1):
    plt.title(f'epoch {fname}; loss: {loss}')
    plt.plot(bins, simulated_g.detach().cpu().numpy(), c='red', label='sim.' )
    plt.plot(bins, target_g, linewidth=2,linestyle='--', c='black', label='exp.')
    plt.legend()
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    # plt.ylim(0 , .6)
    plt.savefig(path + '/' + rdf_title, bbox_inches='tight')
    plt.show()
    plt.close()
def plot_rdfs2(bins, target_g, simulated_g, imp_gt, imp_sim, fname, path, tau, rdf_title, loss=-1):
    
    # plt.plot(bins, simulated_g.detach().cpu().numpy(), c='red', label='sim.' )
    # plt.plot(bins, target_g, linewidth=2,linestyle='--', c='black', label='exp.')
    imp_loss = (imp_gt - imp_sim).pow(2).mean()
    plt.plot(bins, imp_gt.detach().cpu().numpy(), label='imp gt')
    plt.plot(bins, imp_sim.detach().cpu().numpy(), label= 'imp sim' )
    plt.title(f'loss: {loss}, imp_loss: {imp_loss} ')
    plt.legend()
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    # plt.ylim(0 , .6)
    plt.savefig(path + '/' + rdf_title, bbox_inches='tight')
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
    hist_gt = 100*hist_gt/ hist_gt.sum()
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




def fit_rdf(project_name, suggestion_id, device,):
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
    
    num_replicas = 96

    samples = np.random.choice(np.arange(train_dataset.__len__()), num_replicas)
    init_data_arr = [train_dataset.__getitem__(i) for i in samples]


    n_epochs = 100  # number of epochs to train for
    cutoff = 15 # cutoff for interatomic distances (I don't think this is used)
    nbins = 500 # bins for the rdf histogram
    steps = 90
    eq_steps = 10
    tau = steps + eq_steps # this is the number of timesteps, idk why it's called tau
    
    start = 1e-6 # start of rdf range
    end = 10 # end of rdf range
    lr_initial = .0002 # learning rate passed to optim
    dt = 0.5 * units.fs
    temp = 500* units.kB
    ttime =  20   #ttime is only used for NVT setup - it's not the total time
    n_atoms =init_data_arr[0]['pos'].shape[0] 
    targeEkin = 0.5 * (3.0 * n_atoms) * temp
    Q = 3.0 * n_atoms * temp * (ttime * dt)**2
    use_chain = False
    rdf_skip = 10
    checkpoint_epoch = 10
    clip_value = 0.1
    ic_stddev = 0.05
    # rdf_title = f'/replicas_ch{checkpoint_epoch}/rdf_plot_{num_replicas}replicas_{tau}t'
    rdf_title = f'tester'

    reset_each_mega_epoch = True
    mega_epoch_len = 5


    atoms_arr = [data_to_atoms(init_data) for init_data in init_data_arr]
    systems_arr= [System(atoms, device=device) for atoms in atoms_arr]
    [system.set_temperature(temp) for system in systems_arr]
    vels = [system.get_velocities() for system in systems_arr]

    # PURE OVITO STUFF 
    NL = NeighborList(natural_cutoffs(atoms_arr[0]), self_interaction=False)
    NL.update(atoms_arr[0])
    bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T

    
    atom_types_list = list(set(atoms_arr[0].get_chemical_symbols()))
    atom_types = atoms_arr[0].get_chemical_symbols()
    type_to_index = {value: index for index, value in enumerate(atom_types_list)}
    typeid = np.zeros(n_atoms, dtype=int)
    for i, _type in enumerate(atom_types):
        typeid[i] = type_to_index[_type]   

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
    g_obs = find_hr_from_file(MOLECULE, SIZE, n_bins=nbins, start=start, end=end)
    # count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)

    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=lr_initial)

    loss_log = []
    loss_js_log = []
    traj = []

    # Convert `g_obs` to a PyTorch tensor and move it to the same device as `g`
    g_obs_tensor = torch.from_numpy(g_obs).to(device)
    implicit_gt_obs = torch.load('implicit_data/implicit_obs')
    implicit_sim_obs = torch.load('implicit_data/implicit_sim')
    print("DIFFERENCE:", (g_obs_tensor - implicit_gt_obs).pow(2).sum())
    
    print("Training for {} epochs".format(n_epochs))
    # pdb.set_trace()
    curr_init_positions = [system.get_positions() + np.random.normal(0, .05, system.get_positions().shape) for system in systems_arr]
    curr_init_vels = [system.get_velocities() for system in systems_arr]
    for epoch in range(0, n_epochs):
        print("EPOCH: ", epoch)
        current_time = datetime.now() 

        # if you are in the same mega epoch, use the same initial condition
        if epoch == 0 or  epoch % mega_epoch_len != 0:
            [system.set_positions(curr_init_positions[i]) for i, system in enumerate(systems_arr)]
            [system.set_velocities(curr_init_vels[i]) for i, system in enumerate(systems_arr)]
        # if you are in a different mega epoch, you need to 
        elif reset_each_mega_epoch:
            samples = np.random.choice(np.arange(train_dataset.__len__()), num_replicas)
            data_arr = [train_dataset.__getitem__(i) for i in samples]
            for data in data_arr:
                data.pos += torch.normal(torch.zeros_like(data.pos), ic_stddev) 
            [system.set_positions(data_arr[i].pos.cpu().detach().numpy()) for i,  system in enumerate(systems_arr)] 
            [system.set_temperature(temp) for system in systems_arr]
            curr_init_positions = [system.get_positions() for system in systems_arr ]
            curr_init_vels = [system.get_velocities() for system in systems_arr]
        
        restart = epoch != 0 and reset_each_mega_epoch and epoch % mega_epoch_len == 0
        trajs = sim.simulate(steps=tau, frequency=int(tau), dt = dt, restart=True)
        # download_ovito(trajs, dt, bonds, atom_types_list, typeid, ovito_config, epoch*tau)
        
        v_t, q_t, pv_t = trajs
        # _, bins, g = obs(q_t[::rdf_skip, i])
        del(v_t)
        del(pv_t)
        q_t_obs = torch.stack([obs(q_t[eq_steps::rdf_skip, i, : , :])[2] for i in range(q_t.shape[1])])
        #TODO: replace with vmap???

        # Now, let's average over all the rows for g (in the 99 dimension)
        g = q_t_obs.mean(dim=0)


        if not os.path.exists(model_path + '/adjoint_meanRDFs'):
            os.makedirs(model_path + '/adjoint_meanRDFs')
        torch.save(g, model_path + f'/adjoint_meanRDFs/RDFepoch{epoch}')

        loss = (g - g_obs_tensor).pow(2).mean()
        print("LOSS: ", loss.item())
        loss_log.append(loss.item())
        np.save(model_path + '/adjointRDFloss.npy', np.array(loss_log))
        # if  epoch % 25 == 0:
        #    plot_rdfs(xnew, g_obs, g, i, model_path, tau, rdf_title, loss=loss.item())
        #    plot_rdfs2(xnew, g_obs, g, implicit_gt_obs, implicit_sim_obs, epoch, model_path, tau, rdf_title, loss=loss.item())

        # Calculate the loss
        

        results = loss.backward()
        
        duration = (datetime.now() - current_time)
        
        clip_grad_norm_(diffeq.parameters(), clip_value)
        optimizer.step()
        optimizer.zero_grad()
        
        
        if torch.isnan(loss):
            plt.plot(loss_log, list(range(epoch)))
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return np.array(loss_log[-16:-1]).mean()
        else:
            plt.plot(range(len(loss_log)), loss_log, label="adjoint_loss")
            if os.path.isfile(model_path + '/implicit_RDFloss.npy'):
                implicit_loss = np.load(model_path + '/implicit_RDFloss.npy')
                plt.plot(range(len(implicit_loss)), implicit_loss, label="implicit loss")
            plt.yscale("log")
            plt.title("Loss vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
            plt.close()

        # check for loss convergence
        min_idx = np.array(loss_log).argmin()

        if epoch - min_idx >= 125:
            print("converged")
            break
    ovito_config.close()

    # train_traj = [var[1] for var in diffeq.traj]
    # save_traj(system, train_traj, model_path + '/train.xyz', skip=10)

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

def download_ovito(trajs, dt, bonds, atom_types_list, typeid, ovito_config, absolute_step):
        tau = trajs[0].shape[0]
        for i in range(tau):
            radii = trajs[1][i][0]
            velocities = trajs[0][i][0]
            n_atoms = trajs[0].shape[2]
            # Particle positions, velocities, diameter
            partpos = detach_numpy(radii).tolist()
            velocities = detach_numpy(velocities).tolist()
            diameter = (10*0.08*np.ones((n_atoms,))).tolist()
            # Now make gsd file
            s = gsd.hoomd.Frame()
            s.configuration.step = absolute_step + i
            s.particles.N= n_atoms
            s.particles.position = partpos
            s.particles.velocity = velocities
            s.particles.diameter = diameter
            s.configuration.box=[10.0, 10.0, 10.0,0,0,0]
            #extract bond and atom type information
            # pdb.set_trace()
            s.bonds.N =  bonds.shape[0]
            s.bonds.types = atom_types_list
            s.bonds.typeid = typeid
            s.bonds.group = detach_numpy(bonds)
            ovito_config.append(s)


fit_rdf("adj_vs_implicit_final", "naphthalene_100tau", "cuda")
