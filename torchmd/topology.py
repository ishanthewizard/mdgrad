import torch
import numpy as np

def generate_pair_index(N, index_tuple):

    import itertools

    mask_sel = torch.zeros(N, N)

    if index_tuple is not None:
        pair_mask = torch.LongTensor( [list(items) for items in itertools.product(index_tuple[0], 
                                                                                  index_tuple[1])]) 

        #todo: imporse index convention
        mask_sel[pair_mask[:, 0], pair_mask[:, 1]] = 1
        mask_sel[pair_mask[:, 1], pair_mask[:, 0]] = 1
    
    return mask_sel


def generate_nbr_list(xyz, cutoff, cell, index_tuple=None, ex_pairs=None, get_dis=False):
    
    # todo: make it compatible for non-cubic cells
    # todo: topology should be a class to handle some initialization 
    device = xyz.device

    dis_mat = (xyz[..., None, :, :] - xyz[..., :, None, :])

    if index_tuple is not None:
        N = xyz.shape[-2] # the 2nd dim is the atoms dim

        mask_sel = generate_pair_index(N, index_tuple).to(device)
        # todo handle this calculation like a sparse tensor 
        dis_mat =  dis_mat * mask_sel[..., None]

    if ex_pairs is not None:

        N = xyz.shape[-2] # the 2nd dim is the atoms dim

        mask = torch.ones(N, N)
        mask[ex_pairs[:,0], ex_pairs[:, 1]] = 0
        mask[ex_pairs[:,1], ex_pairs[:, 0]] = 0

        # todo handle this calculation like a sparse tensor 
        dis_mat = dis_mat * mask[..., None].to(device)

    offsets = -dis_mat.ge(0.5 *  cell).to(torch.float).to(device) + \
                    dis_mat.lt(-0.5 *  cell).to(torch.float).to(device)
    dis_mat = dis_mat + offsets * cell
    dis_sq = torch.triu( dis_mat.pow(2).sum(-1) )
    mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
    nbr_list = torch.triu(mask.to(torch.long)).nonzero()

    if get_dis:
        return nbr_list, dis_sq[mask].sqrt(), offsets 
    else:
        return nbr_list, offsets

def get_offsets(vecs, cell, device):
    
    offsets = -vecs.ge(0.5 *  cell).to(torch.float).to(device) + \
                vecs.lt(-0.5 *  cell).to(torch.float).to(device)
    
    return offsets


def generate_angle_list(nbr_list):
    '''
        Contributed by saxelrod
    '''

    assert nbr_list.shape[1] == 3 

    nbr_list = make_directed(nbr_list)

    mask = (nbr_list[:, 2, None] == nbr_list[:, 1]) * (
            nbr_list[:, 1, None] != nbr_list[:, 2]) * ( 
            nbr_list[:, None, 0] == nbr_list[:, 0]) # select the same frame 

    third_atoms = nbr_list[:, 2].repeat(nbr_list.shape[0], 1)[mask]
    num_angles = mask.sum(-1)

    nbr_repeats = torch.LongTensor(
                np.repeat(nbr_list.numpy(), num_angles.tolist(), axis=0))

    angle_list = torch.cat(
        (nbr_repeats, third_atoms.reshape(-1, 1)), dim=1)
    
    return angle_list 

def make_directed(nbr_list):
    """
    Check if a neighbor list is directed, and make it
    directed if it isn't. Contributed by saxelrod
    Args:
        nbr_list (torch.LongTensor): neighbor list
    Returns:
        new_nbrs (torch.LongTensor): directed neighbor
            list
        directed (bool): whether the old one was directed
            or not  
    """
    nbr_list_reverse = torch.stack([nbr_list[:,0], nbr_list[:, 2], nbr_list[:,1]], dim=-1)
    new_nbrs = torch.cat([nbr_list, nbr_list_reverse], dim=0)
    
    return new_nbrs