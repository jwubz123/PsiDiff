import os
import pickle
import argparse
import numpy as np
from psikit import Psikit
from tqdm.auto import tqdm
import glob
from pathlib import Path
from rdkit import Chem
from copy import deepcopy

import requests
import time
import traceback

from functools import wraps
from threading import Thread


def set_time_limit(t, start_idx, idx_txt):
    def auto_quit(t1, start_idx1, idx_txt):

        time.sleep(t1)
        print("time out {}".format(t1))
        # f = open(f'{idx_txt}.txt','a') # w : writing mode  /  r : reading mode  /  a  :  appending mode
        # f.write(str(start_idx1) + '\n')
        # f.flush()
        # print('start_idx1: ', start_idx1)
        os._exit(1) 
    def decorator(f):

        @wraps(f)
        def wrapper(*args,**kwargs):
            
            t1=Thread(target=auto_quit,args=(t, start_idx, idx_txt)) 
            t2=Thread(target=f,args=args,kwargs=kwargs)
            t1.setDaemon(True) 
            t1.start()
            t2.start()
            t2.join() 
        return wrapper
    return decorator

class PropertyCalculator(object):

    def __init__(self, threads, memory, seed, time_limit):
        super().__init__()
        self.pk = Psikit(threads=threads, memory=memory)
        self.seed = seed
        self.time_limit = time_limit
    



    def __call__(self, path, data, cache_ref_fn, start_idx, idx_txt):
        print('start index: ', start_idx)
        
        @set_time_limit(self.time_limit, start_idx, idx_txt)
        def cal_energy(pk):
            energy, homo, lumo, dipo = pk.energy(), pk.HOMO, pk.LUMO, pk.dipolemoment[-1]
            data['prop_energy'].append(energy)
            data['prop_homo'].append(homo)
            data['prop_lumo'].append(lumo)
            data['prop_gap'].append(np.abs(homo - lumo))
            data['prop_dipo'].append(dipo)
            with open(cache_ref_fn, 'wb') as f:
                pickle.dump(data, f)
            print('Saved', pdb_id)
            # return energy, homo, lumo, dipo
            return
        
        pdb_id = data['pdb_id']


        data['prop_energy'] = []
        data['prop_homo'] = []
        data['prop_lumo'] = []
        data['prop_dipo'] = []
        data['prop_gap'] = []

        suppl = Chem.SDMolSupplier(path + f'/{pdb_id}/{pdb_id}_ligand.sdf', sanitize=False)

        mol = suppl[0]
        smiles = Chem.MolToSmiles(mol)
        print(pdb_id, smiles)
        try:
            self.pk.read_from_smiles(smiles)
        except:
            self.pk.mol = mol

        try:

            cal_energy(self.pk)
            
        except:
            print('Failed', pdb_id)
            print(traceback.format_exc())

            pass
        
            
        return data
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--threads', type=int, default=14)
    parser.add_argument('--memory', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--time_limit', type=int, default=500)
    parser.add_argument('--idx_txt', type=str, default='test')
    parser.add_argument('--txt', type=str, default='train')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    args = parser.parse_args()
    
    raw_path = 'datasets'
    prop_cal = PropertyCalculator(threads=args.threads, memory=args.memory, seed=args.seed, time_limit=args.time_limit)
    with open(f'../{args.txt}_list.txt', 'r') as f:
        pdb_id_list = f.read().split()
        f.close()
    pdb_id_list = pdb_id_list[args.start:args.end]
    print(args.txt, args.start, args.end)
    failed_pdb = []
        
    try:
        start_file = open(f'{args.idx_txt}.txt','r')
        start_idx = start_file.read().split()
        start_file.close()
        start_idx = int(start_idx[-1])
    except:
        start_idx = 0
    pdb_id_list = pdb_id_list[start_idx:args.end]
    
    print('length of pdb_id_list: ', len(pdb_id_list))
    for pdb_id in pdb_id_list: 
        cache_ref_fn = f'{args.dataset}/{pdb_id}.pkl'
        start_idx += 1
        f = open(f'{args.idx_txt}.txt','a') 
        f.write(str(start_idx) + '\n')
        f.flush()
        print('start: ', start_idx)
        if not os.path.exists(cache_ref_fn):
            # try:    
            print('ref ', pdb_id)
            data = {}
            data['pdb_id'] = pdb_id
            dset_prop = None

            dset_prop = prop_cal(raw_path, data, cache_ref_fn, start_idx, args.idx_txt)

        else:   
            print('Exists', pdb_id)
    print(failed_pdb)