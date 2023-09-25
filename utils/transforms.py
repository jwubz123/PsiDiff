import torch
from torch_geometric.data import DataLoader

def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]

class BatchDownSamplingIndex(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        lig, tar, pdb_id = data
        max_prev_batch = 0
        max_prev_batch_complex = 0
        idx = []
        complex_idx = []
        target_counts = torch.unique(tar.gen_xyz_p_batch, return_counts=True)[1]
        complex_counts = torch.unique(lig.complex_pos_batch, return_counts=True)[1]
        ligand_counts = complex_counts - target_counts
        for i in range(len(tar.target_idx_ptr)-1):

            target_idx = tar.target_idx[tar.target_idx_ptr[i]:tar.target_idx_ptr[i+1]]
            idx.append(target_idx + max_prev_batch)
            complex_idx.append(max_prev_batch_complex + torch.arange(ligand_counts[i], device=ligand_counts.device))
            complex_idx.append(target_idx + max_prev_batch_complex + ligand_counts[i])
            max_prev_batch += target_counts[i]
            max_prev_batch_complex += complex_counts[i]
        idx = torch.cat(idx)
        complex_idx = torch.cat(complex_idx)
        tar.target_idx = idx
        tar.complex_idx = complex_idx
        data = (lig, tar, pdb_id)
        return data

    