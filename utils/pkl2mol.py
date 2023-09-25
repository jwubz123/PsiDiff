import torch
from copy import deepcopy
import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from pathlib import Path
import glob
import pickle
import numpy as np

def pkl2mol(ligand, pdb_id, real_root, save_root, try_num=0, do_ff=False):
    if try_num == 0:
        mol = Chem.SDMolSupplier(f'{real_root}/{pdb_id}/{pdb_id}_ligand.sdf', sanitize=False)
        m = mol[0]
    else: 
        m = Chem.MolFromMol2File(f'{real_root}/{pdb_id}/{pdb_id}_ligand.mol2',
                                        sanitize=False, cleanupSubstructures=False, removeHs=False)
    
    em = Chem.EditableMol(m)
    atoms_to_remove = []
    for i, atom in enumerate(m.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            atoms_to_remove.append(i)
    atoms_to_remove.sort(reverse=True)
    for atom in atoms_to_remove:
        em.RemoveAtom(atom)
    m = em.GetMol()

    for i in range(len(ligand['gen_pos'])):
        gen_mol = set_rdmol_positions(m, ligand['gen_pos'][i])
        Chem.MolToMolFile(gen_mol, f'{save_root}/gen_mol/{pdb_id}_{i}.mol', kekulize=False)
        if do_ff:
            ff_mol = Chem.MolFromMolFile(f'{save_root}/gen_mol/{pdb_id}_{i}.mol') 
            MMFFOptimizeMolecule(ff_mol)
            Chem.MolToMolFile(ff_mol, f'{save_root}/ff_mol/{pdb_id}_{i}.mol', kekulize=False)
def mol2tomol(ligand, pdb_id, real_root, save_root, try_num=0):
    if try_num == 0:
        mol = Chem.SDMolSupplier(f'{real_root}/{pdb_id}/{pdb_id}_ligand.sdf', sanitize=False)
        m = mol[0]
    else: 
        m = Chem.MolFromMol2File(f'{real_root}/{pdb_id}/{pdb_id}_ligand.mol2',
                                        sanitize=False, cleanupSubstructures=False, removeHs=False)
    
    em = Chem.EditableMol(m)
    atoms_to_remove = []
    for i, atom in enumerate(m.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            atoms_to_remove.append(i)
    atoms_to_remove.sort(reverse=True)
    for atom in atoms_to_remove:
        em.RemoveAtom(atom)
    m = em.GetMol()
    raw_pos = m.GetConformer().GetPositions()
    raw_mol = set_rdmol_positions(m, raw_pos)
    Chem.MolToMolFile(raw_mol, f'{save_root}/ref_mol/{pdb_id}_ref.mol', kekulize=False)

def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol

if __name__ == '__main__':

    # Logging
    path = '/shared_space/jiamin/generation/DiffBindComplexCond/logs/evo_cat_dir_no_dist_embed_2023_01_05__14_46_15_resume/sample_2023_02_24__12_02_33_no_guidance_debug'
    real_root = '/shared_space/xtalpi_lab/Datasets/PDBbind/PDBbind_v2020/v2020_all'
    save_root = path
    Path(save_root).mkdir(exist_ok=True)  
    Path(f'{save_root}/gen_mol').mkdir(exist_ok=True) 
    do_ff = True
    if do_ff:
        Path(f'{save_root}/ff_mol').mkdir(exist_ok=True) 
    
    # Load results
    
    results = {}
    
    best_rmsd_list = []
    rmsd_list = []
    centroid_dist_list = []

    ff_best_rmsd_list = []
    ff_rmsd_list = []
    ff_centroid_dist_list = []

    pdb_list = []
    # with open('../test_list.txt', 'r') as f:
    #     pdb_list = f.read().split()
    for p in glob.glob(f'{path}/*.pkl'):
    # for pdb_id in pdb_list:
        pdb_id = Path(p).stem[-4:]
        with open(f'{path}/20_samples_{pdb_id}.pkl', 'rb') as f:
            ligand = pickle.load(f)
            f.close()
        # ligand=0

        # Evaluator
        # best_rmsd, rmsd, ff_best_rmsd, ff_rmsd = save_mol(ligand, pdb_id, real_root, save_root=save_root)
        try:
            pkl2mol(ligand, pdb_id, real_root, save_root=save_root, try_num=0, do_ff=do_ff)
        except:
            pkl2mol(ligand, pdb_id, real_root, save_root=save_root, try_num=1, do_ff=do_ff)
