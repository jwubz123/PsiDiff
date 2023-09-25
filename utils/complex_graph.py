from torch_geometric.data import Dataset
import torch
import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize     

class ComplexDataset(Dataset):

    def __init__(self, id_idx, path, save_charge=True):
        super().__init__()
        self.save_charge = save_charge
        self.prop_root = 'PDBBind20_property'
        self.real_root = 'datasets'
        self.id_idx = id_idx
        self.data = torch.load(path)
        self.pos = self._pos()
        self.addnorm = self._addnorm()
        self._save_prop()

    def __getitem__(self, idx):

        data = self.data[idx]
        return data

    def __len__(self):
        return len(self.data)
    
    def _pos(self):
        """complex coordinates."""
        for data in self.data:
            data[0].complex_pos = torch.cat([data[0].pos, data[1].gen_xyz_p], dim=0)
        return data
    
    def _addnorm(self):
        """add normalized ligand and target coordinates"""
        for data in self.data:
            pos_norm, gen_xyz_p_norm, atom_coords_p_norm, mu, std = self.normalize_pos_std(data)
            # pos_norm, gen_xyz_p_norm, atom_coords_p_norm, mu, std = self.normalize_pos_neg1to1(data)
            data[0].pos = pos_norm
            data[1].gen_xyz_p = gen_xyz_p_norm
            data[1].atom_coords_p = atom_coords_p_norm
            data[1].mu = mu
            data[1].std = std
            data[0].complex_pos = torch.cat([data[0].pos, data[1].gen_xyz_p], dim=0)
        return data
    
    def normalize_pos_std(self, data):
        ligand = data[0]
        target = data[1]
        atom_coords_p_norm = torch.zeros_like(target.atom_coords_p)
        pos_norm = torch.zeros_like(ligand.pos)
        gen_xyz_p_norm = torch.zeros_like(target.gen_xyz_p)

        mu = torch.mean(target.atom_coords_p, dim=0)

        std_max = 0
        for j in range(3):
            std = torch.std(target.atom_coords_p[:,j])
            if std > std_max:
                std_max = std
        std = std_max
        
        for i in range(3):
            atom_coords_p_norm[:, i] = (target.atom_coords_p[:, i] - mu[i]) / std
            pos_norm[:, i] = (ligand.pos[:, i] - mu[i]) / std
            gen_xyz_p_norm[:, i] = (target.gen_xyz_p[:, i] - mu[i]) / std

        return pos_norm, gen_xyz_p_norm, atom_coords_p_norm, mu, std
    def normalize_pos_neg1to1(self, data):
        ligand = data[0]
        target = data[1]
        atom_coords_p_norm = torch.zeros_like(target.atom_coords_p)
        pos_norm = torch.zeros_like(ligand.pos)
        gen_xyz_p_norm = torch.zeros_like(target.gen_xyz_p)

        mu = target.atom_coords_p.min()
        std_max = 0
        for i in range(3):
            std = target.atom_coords_p[:, i].max() - target.atom_coords_p[:, i].min()
            if std > std_max:
                std_max = std
        std = std_max

        atom_coords_p_norm = 2 * (target.atom_coords_p - mu) / std - 1
        pos_norm = 2 * (ligand.pos - mu) / std - 1
        gen_xyz_p_norm = 2 * (target.gen_xyz_p - mu) / std - 1

        return pos_norm, gen_xyz_p_norm, atom_coords_p_norm, mu, std
    def _save_prop(self):
        new_list = []
        for data in self.data:
            if data[2] in self.id_idx:
                pdb_id = data[2]
                ### load calculated_properties
                with open(f'{self.prop_root}/{data[2]}.pkl', 'rb') as f:
                    dset = pickle.load(f)
                f.close()
                # ### Normalize properties
                # for key, item in dset:
                #     dset[key] == normalized_prop(item)

                num_atoms = data[0].x.shape[0]
                if self.save_charge:
                    ### load charges
                    try:
                        mol = Chem.MolFromMol2File(self.real_root + f'/{pdb_id}/{pdb_id}_ligand.mol2')
                        mol = remove_atoms(mol)
                        AllChem.ComputeGasteigerCharges(mol)
                    except:
                        suppl = Chem.SDMolSupplier(self.real_root + f'/{pdb_id}/{pdb_id}_ligand.sdf', sanitize=False)
                        mol = suppl[0]
                        mol = remove_atoms(mol)
                        AllChem.ComputeGasteigerCharges(mol)
                    charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
                    charges = torch.tensor(charges)
                    if torch.isinf(charges).any() or torch.isnan(charges).any():
                        charges = cal_charge_remove_salt(mol)
                        charges = torch.tensor(charges)
                        if torch.isinf(charges).any() or torch.isnan(charges).any():
                            print('nan in ', pdb_id)
                            continue
                    if charges.shape[0] != data[0].x.shape[0]:
                        print('Missing charges in ', pdb_id)
                        continue
                    data[0].atom_charges = charges
                
                ### properties to a dict
                data[0].num_atoms = num_atoms
                data[0].prop_energy = dset['prop_energy'][0]
                data[0].prop_gap = dset['prop_gap'][0]
                new_list.append(data)
            else: 
                continue
        self.data = new_list

def cal_charge_remove_salt(mol):                                                                        
    salt_remover = rdMolStandardize.FragmentRemover()                                                                                                                                                                                                                                                                       
    parent = salt_remover.remove(mol)                                                                                                                                                                            
    assert (mol.GetNumAtoms() == len(parent.GetAtoms()))
    Chem.SanitizeMol(parent)                                                                                                                                                                                   
    Chem.rdPartialCharges.ComputeGasteigerCharges(parent)                                                                                                                                                           
    charges = [(at.GetDoubleProp('_GasteigerCharge')+at.GetDoubleProp('_GasteigerHCharge')) for at in parent.GetAtoms()]
    return charges

def remove_atoms(m):
    em = Chem.EditableMol(m)
    atoms_to_remove = []
    for i, atom in enumerate(m.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            atoms_to_remove.append(i)
    atoms_to_remove.sort(reverse=True)
    for atom in atoms_to_remove:
        em.RemoveAtom(atom)
    m = em.GetMol()
    return m
      
    
if __name__ == '__main__':
    surfix_list = ['train', 'val', 'test'] #  
    len_saved = []
    len_list = []
    save_charge = False
    if save_charge:
        with_charge = 'with'
    else:
        with_charge = 'no'
    for surfix in surfix_list:
        idx_file = open(f'saved_{surfix}_list.txt','r')
        idx_list = idx_file.read().split()
        data_root = f'datasets/equibind_data_pocket/{surfix}.pt'
        save_root = f'datasets/equibind_data_pocket_add_std_norm_prop/{surfix}_{with_charge}_charges.pt'
        dataset = ComplexDataset(idx_list, data_root, save_charge)
        torch.save(dataset, save_root)
        len_list.append(len(idx_list))
        len_saved.append(len(dataset))
        print('Data Saved at ', save_root)
    print(f'Length of {surfix_list} list ', len_list)
    print(f'Length of {surfix_list} data ', len_saved)
