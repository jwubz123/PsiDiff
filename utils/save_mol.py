import torch
from copy import deepcopy
import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

def save_mol(ligand, pdb_id, real_root, save_root, try_num=0, cal_ff=True):
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
    
    raw_pos = torch.tensor(m.GetConformer().GetPositions(), device=ligand['gen_pos'].device)
    m.GetNumAtoms()
     
    assert m.GetNumAtoms() == ligand['gen_pos'].size(0)
    
    best_rmsd, aligned_mol, gen_mol = cal_aligned_rmsd(m, ligand['gen_pos'])
    rmsd = cal_rmsd(raw_pos, ligand['gen_pos'])
    centroid_dist = cal_centroid_dist(raw_pos, ligand['gen_pos'])
    
    Chem.MolToMolFile(gen_mol, f'{save_root}/gen_mol/{pdb_id}_gen.mol', kekulize=False)
    Chem.MolToMolFile(aligned_mol, f'{save_root}/aligned_gen_mol/{pdb_id}_gen_aligned.mol', kekulize=False)
    if cal_ff:
        ff_mol = Chem.MolFromMolFile(f'{save_root}/gen_mol/{pdb_id}_gen.mol') 
        MMFFOptimizeMolecule(ff_mol)

        ff_pos = torch.tensor(ff_mol.GetConformer().GetPositions(), device=ligand['gen_pos'].device)

        ff_rmsd = cal_rmsd(raw_pos, ff_pos)
        ff_centroid_dist = cal_centroid_dist(raw_pos, ff_pos)
        ff_best_rmsd, ff_aligned_mol, ff_mol = cal_aligned_rmsd(m, ff_pos)
        
        
        Chem.MolToMolFile(ff_mol, f'{save_root}/ff_mol/{pdb_id}_gen_ff.mol', kekulize=False)
        Chem.MolToMolFile(ff_aligned_mol, f'{save_root}/ff_aligned_mol/{pdb_id}_gen_ff_aligned.mol', kekulize=False)
    else: 
        ff_best_rmsd = best_rmsd
        ff_rmsd = rmsd
        ff_centroid_dist = centroid_dist
    return best_rmsd, rmsd, centroid_dist, ff_best_rmsd, ff_rmsd, ff_centroid_dist
    

def cal_rmsd(pos1, pos2, device='cpu'):
    N = pos1.size(0)
    rmsd = torch.sqrt(1 / N * torch.sum((pos1.to(device) - pos2.to(device)) ** 2))
    return rmsd

def cal_aligned_rmsd(ref_mol, gen_pos):
    gen_mol = pos_to_mol(gen_pos, ref_mol)
    aligned_mol= deepcopy(gen_mol)
    best_rmsd = rdMolAlign.GetBestRMS(aligned_mol, ref_mol)
    # rmsd_aligned = rdMolAlign.AlignMol(aligned_mol, ref_mol)
    return best_rmsd, aligned_mol, gen_mol

def pos_to_mol(pos, ref_mol):
    noHidx = [idx for idx in range(ref_mol.GetNumAtoms()) if ref_mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]
    gen_mol = deepcopy(ref_mol)
    conf_gen = gen_mol.GetConformer()
    xyz_coordinates_gen = pos

    for i in range(gen_mol.GetNumAtoms()):
        x,y,z = xyz_coordinates_gen[i]
        x,y,z = x.double().item(), y.double().item(), z.double().item()
        conf_gen.SetAtomPosition(i,Point3D(x,y,z))
    return gen_mol

def cal_centroid_dist(pos1, pos2, device='cpu'):
    pos1_center = torch.mean(pos1, dim=0, keepdim=True).to(device).double()
    pos2_center = torch.mean(pos2, dim=0, keepdim=True).to(device).double()
    dist = torch.cdist(pos1_center, pos2_center)
    return dist

    
