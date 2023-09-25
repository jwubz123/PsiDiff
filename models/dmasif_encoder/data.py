import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from scipy.spatial.transform import Rotation
import tarfile
from pathlib import Path
# from data_preprocessing.convert_pdb2npy import convert_pdbs

tensor = torch.FloatTensor
inttensor = torch.LongTensor

def numpy(x):
    return x.detach().cpu().numpy()

class CenterAtoms(object):
    r"""Centers a protein"""

    def __call__(self, data):
        atom_center1 = data.atom_coords_p.mean(dim=-2, keepdim=True)

        data.atom_coords_p = data.atom_coords_p - atom_center1
        data.atom_center1 = atom_center1
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

def load_protein_npy(pdb_id, data_dir, center=False):
    """Loads a protein surface mesh and its features"""

    atom_coords = tensor(np.load(data_dir / (pdb_id + "_pocket_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (pdb_id + "_pocket_atomtypes.npy")))

    protein_data = Data(
        num_nodes=None,
        y=None,
        atom_coords=atom_coords,
        atom_types=atom_types,
    )
    return protein_data


class SingleData(Data):
    def __init__(
        self,
        xyz_p=None,
        chemical_features_p=None,
        normals_p=None,
        center_location_p=None,
        atom_coords_p=None,
        atom_types_p=None,
        atom_center1=None
    ):
        super().__init__()
        self.xyz_p = xyz_p
        self.chemical_features_p = chemical_features_p
        self.normals_p = normals_p
        self.center_location_p = center_location_p
        self.atom_coords_p = atom_coords_p
        self.atom_types_p = atom_types_p
        self.atom_center1 = atom_center1

    def __inc__(self, key, value, *args, **kwargs):
       
        return super(SingleData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if ("index" in key) or ("face" in key):
            return 1
        else:
            return 0

def load_protein_single(pdb_id, data_dir):
    """Loads a protein surface mesh and its features"""
    p_id = pdb_id
    p = load_protein_npy(p_id, data_dir, center=False)

    protein_single_data = SingleData(
        atom_coords_p=p["atom_coords"],
        atom_types_p=p["atom_types"]
    )
    return protein_single_data


class ProteinSurfaces(InMemoryDataset):
    url = ""

    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(ProteinSurfaces, self).__init__(root, transform, pre_transform)
        path = f'{self.root}/processed/training_data.pt' if train else f'{self.root}/processed/testing_data.pt'
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return "pdbbind_v2019_refined.tar.gz"

    @property
    def processed_file_names(self):

        file_names = [
            "training_data.pt",
            "testing_data.pt",
            "training_data_ids.npy",
            "testing_data_ids.npy",
        ]
        return file_names



    def process(self):
        
        ### Choose one from the following:
        
        ### 1. for dataset refine:
        pdb_dir_str = f'{self.root}/raw/refined-set'
        pdb_dir = Path(f'{self.root}') / "raw" / "refined-set"
        protein_dir = Path(f'{self.root}') / "raw" / "refined-set_npy"
        data_dir = '../datasets/pdbbind_v2019_refined.tar.gz'
        lists_dir = Path(f'{self.root}/lists')
        # ### 2. for dataset others:
        # pdb_dir_str = f'{self.root}/raw/v2019-other-PL'
        # pdb_dir = Path(f'{self.root}') / "raw" / "v2019-other-PL"
        # protein_dir = Path(f'{self.root}') / "raw" / "others-set_npy"
        # data_dir = '../datasets/pdbbind_v2019_other_PL.tar.gz'
        # lists_dir = Path(f'{self.root}/lists')
        
        # Untar surface files
        if not pdb_dir.exists():
            print('untaring datasets to get pdb files!')
            tar = tarfile.open(data_dir)
            tar.extractall(self.raw_dir)
            tar.close()

        if not protein_dir.exists():
            protein_dir.mkdir(parents=False, exist_ok=False)
            convert_pdbs(pdb_dir_str,protein_dir)

        with open(lists_dir / "training.txt") as f_tr, open(
            lists_dir / "testing.txt"
        ) as f_ts:
            training_list = sorted(f_tr.read().splitlines())
            testing_list = sorted(f_ts.read().splitlines())
        
        # # Read data into huge `Data` list.
        training_data = []
        training_data_ids = []
        
        for p in training_list:
            try:
                protein_single = load_protein_single(p, protein_dir)
            except FileNotFoundError:
                continue
            training_data.append(protein_single)
            training_data_ids.append(p)
        testing_data = []
        testing_data_ids = []
        for p in testing_list:
            try:
                protein_single = load_protein_single(p, protein_dir)
            except FileNotFoundError:
                continue
            testing_data.append(protein_single)
            testing_data_ids.append(p)
        if self.pre_filter is not None:
            training_data = [
                data for data in training_data if self.pre_filter(data)
            ]
            testing_data = [
                data for data in testing_data if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            training_data = [
                self.pre_transform(data) for data in training_data
            ]
            testing_data = [
                self.pre_transform(data) for data in testing_data
            ]
        print('len(training_data)', len(training_data))
        training_data, training_slices = self.collate(training_data)
        torch.save(
            (training_data, training_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[2], training_data_ids)
        testing_data, testing_slices = self.collate(testing_data)
        torch.save((testing_data, testing_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_data_ids)

class ProteinSurfacesCasf(InMemoryDataset):
    url = ""

    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(ProteinSurfacesCasf, self).__init__(root, transform, pre_transform)
        path = f'{self.root}/processed/casf_data.pt'
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        file_names = [
        "casf_data.pt",
        "casf_data_ids.npy",
        ]
        return file_names



    def process(self):

        pdb_dir_str = '../datasets/CASF-2016/coreset'
        pdb_dir = Path(pdb_dir_str)
        protein_dir = Path(f'{pdb_dir_str}_npy_32_atoms')
        print(protein_dir)
        lists_dir = Path(f'{self.root}/lists')
        
        with open(lists_dir / "casf.txt") as f_tr:
            training_list = sorted(f_tr.read().splitlines())
        
        # # Read data into huge `Data` list.
        training_data = []
        training_data_ids = []
        
        for p in training_list:
            try:
                protein_single = load_protein_single(p, protein_dir)
            except FileNotFoundError:
                continue
            training_data.append(protein_single)
            training_data_ids.append(p)
        
        print('len(training_data)', training_data[0])
        training_data, training_slices = self.collate(training_data)
        torch.save(
            (training_data, training_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[1], training_data_ids)