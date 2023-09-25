import torch
import numpy as np
from models.dmasif_encoder.helper import *
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from models.dmasif_encoder.helper import numpy, diagonal_ranges
import time


def process_single(protein_single):
    """Turn the PyG data object into a dict."""

    P = {}
    preprocessed = "gen_xyz_p" in protein_single.keys
    # Atom information:
    P["atoms"] = protein_single.atom_coords_p
    P["batch_atoms"] = protein_single.atom_coords_p_batch

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = protein_single.atom_coords_p
    P["atomtypes"] = protein_single.atom_types_p

    P["xyz"] = protein_single.gen_xyz_p if preprocessed else None
    P["normals"] = protein_single.gen_normals_p if preprocessed else None
    P["batch"] = protein_single.gen_batch_p if preprocessed else None
    return P


def process(protein_single, net):
    P1 = process_single(protein_single)
    if not "gen_xyz_p" in protein_single.keys:
        net.preprocess_surface(P1)

    return P1

def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["batch"] == number
    batch_atoms = P_batch["batch_atoms"] == number

    P["batch"] = P_batch["batch"][batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][batch]
    P["normals"] = P_batch["normals"][batch]

    # Atom information:
    P["atoms"] = P_batch["atoms"][batch_atoms]
    P["batch_atoms"] = P_batch["batch_atoms"][batch_atoms]

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = P_batch["atom_xyz"][batch_atoms]
    P["atomtypes"] = P_batch["atomtypes"][batch_atoms]

    return P

def iterate(
    net,
    dataset,
    args,
    device='cpu'
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""
    

    protein_single = dataset

    protein_single.to(device)

    # Generate the surface:
    # torch.cuda.synchronize()
    surface_time = time.time()
    P_batch = process(protein_single, net)
    # torch.cuda.synchronize()
    surface_time = time.time() - surface_time
    outputs = net(P_batch)

    return outputs

def iterate_surface_precompute(dataset, sur_dir, net, data_type, args, device):
    pre_compute_dir = f'{sur_dir}/surface_precompute'
    pre_compute_file_dir = f'{pre_compute_dir}/{data_type}.pt'
    if not (Path(pre_compute_dir).exists()):
        Path(pre_compute_dir).mkdir(parents=False, exist_ok=False)
    if not (Path(pre_compute_file_dir).is_file()):
        processed_dataset = []
        for it, protein_single in enumerate(tqdm(dataset)):
            protein_single.to(device)
            P1 = process(protein_single, net)
            protein_single = protein_single.to_data_list()[0]
            protein_single.gen_xyz_p = P1["xyz"]
            protein_single.gen_normals_p = P1["normals"]
            protein_single.gen_batch_p = P1["batch"]
            processed_dataset.append(protein_single.to("cpu"))
        torch.save(processed_dataset, pre_compute_file_dir)
        print(f'surface_precompute of {data_type} saved!')
    else:
        print(f'surface_precompute of {data_type} exists, loading it')
        processed_dataset = torch.load(pre_compute_file_dir)
    return processed_dataset
