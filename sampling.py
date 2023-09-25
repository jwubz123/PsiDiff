import os
import argparse
import pickle
import yaml
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
from pathlib import Path

from utils.transforms import BatchDownSamplingIndex
from utils.complex_graph import ComplexDataset
from utils.datasets import *
from utils.misc import *

from models.s_theta_net import *
from models.g_phi_net.models import EGNN_energy
from models.g_phi_net.en_diffusion import EnergyDiffusion
from models.s_theta_net.sampling import SamplingWithGuidance


def get_energy_model(config, args):
    #in_node_nf: the numbder of atom type = 2 if not use atom type in the energy function
    in_node_nf = config.in_node_nf
    if config.condition_time:
        in_node_nf = in_node_nf + 1
    if args.guidance_cond == 'multi':
        context_node_nf = len(config.condition_prop)
    elif args.guidance_cond == 'each':
        context_node_nf = 1
    elif args.guidance_cond == 'no_cond':
        context_node_nf = 0
    guidance1 = None

    if len(config.guidance_prop) >= 1:
        if config.guidance_prop[0] == 'atom_charges':
            out_dim = config.max_num_atoms
        else:
            out_dim = 1
        net_energy1 = EGNN_energy(
            in_node_nf=in_node_nf, context_node_nf=context_node_nf,
            n_dims=3, device=args.device, hidden_nf=config.hidden_nf,
            act_fn=torch.nn.SiLU(), n_layers=config.n_layers,
            attention=config.attention, tanh=config.tanh, mode=config.mode, norm_constant=config.norm_constant,
            inv_sublayers=config.inv_sublayers, sin_embedding=config.sin_embedding,
            normalization_factor=config.normalize_factors, aggregation_method=config.aggregation_method,
            out_dim=out_dim)
        
        guidance1 = EnergyDiffusion(
                dynamics=net_energy1,
                in_node_nf=in_node_nf,
                n_dims=3,
                timesteps=config.diffusion_steps,
                noise_schedule=config.diffusion_noise_schedule,
                noise_precision=config.diffusion_noise_precision,
                norm_values=config.normalization_factor,
                include_charges=config.include_charges
            )
    guidance2 = None
    if len(config.guidance_prop) >= 2:
        if config.guidance_prop[1] == 'atom_charges':
            out_dim = config.max_num_atoms
        else:
            out_dim = 1
        net_energy2 = EGNN_energy(
            in_node_nf=in_node_nf, context_node_nf=context_node_nf,
            n_dims=3, device=args.device, hidden_nf=config.hidden_nf,
            act_fn=torch.nn.SiLU(), n_layers=config.n_layers,
            attention=config.attention, tanh=config.tanh, mode=config.mode, norm_constant=config.norm_constant,
            inv_sublayers=config.inv_sublayers, sin_embedding=config.sin_embedding,
            normalization_factor=config.normalize_factors, aggregation_method=config.aggregation_method,
        out_dim=out_dim)
        
        guidance2 = EnergyDiffusion(
                dynamics=net_energy2,
                in_node_nf=in_node_nf,
                n_dims=3,
                timesteps=config.diffusion_steps,
                noise_schedule=config.diffusion_noise_schedule,
                noise_precision=config.diffusion_noise_precision,
                norm_values=config.normalization_factor,
                include_charges=config.include_charges
            )
    guidance3 = None
    if len(config.guidance_prop) == 3:
        if config.guidance_prop[2] == 'atom_charges':
            out_dim = config.max_num_atoms
        else:
            out_dim = 1
        net_energy3 = EGNN_energy(
            in_node_nf=in_node_nf, context_node_nf=context_node_nf,
            n_dims=3, device=args.device, hidden_nf=config.hidden_nf,
            act_fn=torch.nn.SiLU(), n_layers=config.n_layers,
            attention=config.attention, tanh=config.tanh, mode=config.mode, norm_constant=config.norm_constant,
            inv_sublayers=config.inv_sublayers, sin_embedding=config.sin_embedding,
            normalization_factor=config.normalize_factors, aggregation_method=config.aggregation_method,
        out_dim=out_dim)
        
        guidance3 = EnergyDiffusion(
                dynamics=net_energy3,
                in_node_nf=in_node_nf,
                n_dims=3,
                timesteps=config.diffusion_steps,
                noise_schedule=config.diffusion_noise_schedule,
                noise_precision=config.diffusion_noise_precision,
                norm_values=config.normalization_factor,
                include_charges=config.include_charges
            )        
    return guidance1, guidance2, guidance3


def load_checkpoint(config, args):
    ckpt = torch.load(args.ckpt, map_location='cpu')
    gen_model = get_model(config.model).to(args.device)
    gen_model.load_state_dict(ckpt['model'])
    gen_model.eval()
    guidance1, guidance2, guidance3 = get_energy_model(config.model.energy_model, args)
    if guidance1 is not None:
        energy_state_dict1 = torch.load(args.guidance_path1, map_location='cpu')
        guidance1.load_state_dict(energy_state_dict1['model'])
        guidance1.to(args.device)
    if guidance2 is not None:
        energy_state_dict2 = torch.load(args.guidance_path2, map_location='cpu')
        guidance2.load_state_dict(energy_state_dict2['model'])
        guidance2.to(args.device)
    if guidance3 is not None:
        energy_state_dict3 = torch.load(args.guidance_path3, map_location='cpu')
        guidance3.load_state_dict(energy_state_dict3['model'])
        guidance3.to(args.device)
    return gen_model, guidance1, guidance2, guidance3

def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()

def save_pkl(ligand, pdbid, save_root):
    i = 0
    for pdb_id in pdbid:
        result = {}
        result['pdb_id'] = pdb_id
        save_path = os.path.join(save_root, 'samples_%s.pkl' % pdb_id)
        logger.info('Saving samples to: %s' % save_path)

        result['gen_pos'] = ligand.complex_pos[ligand.ptr[i]:ligand.ptr[i + 1]]       
        i += 1
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
            f.close()
        
def sampling_main(args, config, done_pdb, model, db_complex_test, batch_size, start_batch_size, logger, skip):
    test_set_selected = []
    for i, data in enumerate(db_complex_test):
        if not (args.start_idx <= i < args.end_idx): 
            continue
        elif data[2] in done_pdb:
            # logger.info('Molecule#%s is already done.' % data[2])
            continue
        else:
            test_set_selected.append(data)
    if skip:
        removed = test_set_selected.pop(0)
        done_pdb.append(removed[2])
    logger.info('Complexes in test set: %d' % len(test_set_selected))
            
    batch_vars = ["gen_xyz_p", "atom_coords_p", 'complex_pos', 'target_idx']
    transform = BatchDownSamplingIndex()
    if batch_size < start_batch_size:
        test_loader = DataLoader(
            test_set_selected[:batch_size], batch_size=batch_size, follow_batch=batch_vars, shuffle=False, num_workers=2
        )
    else:
        test_loader = DataLoader(
            test_set_selected, batch_size=batch_size, follow_batch=batch_vars, shuffle=False, num_workers=2
        )

    for i, batch in enumerate(tqdm(test_loader)):

        batch = transform(batch)
        ligand, target, pdbid = batch

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_raw = ligand.pos
                ligand, target = ligand.to(args.device), target.to(args.device)
                N = ligand.pos.size(0)
                # pos_init = torch.randn(N, 3).to(args.device)
                
                pos_gen = sampling_model.sampling(
                    ligand,
                    target
                )
                ligand.complex_pos = pos_gen.detach().cpu()

                save_pkl(ligand.cpu(), pdbid, output_dir)
                done_pdb.extend(pdbid)

                break   # No errors occured, break the retry loop
            
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')
        
    return done_pdb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('guidance_path1', type=str, help='path for loading the energy model checkpoint')
    parser.add_argument('--guidance_path2', type=str, default=None, help='path for loading the energy model checkpoint')
    parser.add_argument('--guidance_path3', type=str, default=None, help='path for loading the energy model checkpoint')
    parser.add_argument('--guidance_cond', type=str, default='multi', choices=['multi', 'each', 'no_cond'] ,help='contexts for energy model to condition on')
    parser.add_argument('--save_traj', action='store_true', default=False,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--config_name', type=str, help='path for loading the checkpoint')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--try_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=349)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--clip_local', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=4999,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    
    args = parser.parse_args()
    if args.gpu_id != 0:
        torch.cuda.set_device(args.gpu_id)
            
    # logging
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), f'{args.config_name}.yml'))[0]
    print(config_path)
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(args.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    
    # Logging
    done_pdb = []
    
    if args.resume is not None:
        output_dir = args.resume
        for p in glob(f'{output_dir}/samples_*.pkl'):
            done_pdb.append(Path(p).stem[-4:])
    else:
        output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)
    logger.info(config)
    
    # torch.set_grad_enabled(False)
    # Datasets and loaders
    logger.info('Loading datasets...')
    # db_complex_test = torch.load(config.dataset.test)
    db_complex_test = torch.load(args.test_set)
    pdbIDs_test = [db_complex_test[i][2] for i in range(len(db_complex_test))]
    
    # Model
    logger.info('Loading model...')
    gen_model, guidance1, guidance2, guidance3 = load_checkpoint(config, args)
    sampling_model = SamplingWithGuidance(
                    config.model, 
                    gen_model, 
                    guidance1,
                    guidance2,
                    guidance3,
                    cond_type=args.guidance_cond,
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=args.clip_local,
                    extend_order=config.model.edge_order, 
                    extend_radius=True)
    sampling_model.to(args.device)

    if args.batch_size == 0:
        start_batch_size = config.train.batch_size
    else:
        start_batch_size = args.batch_size
    batch_size = start_batch_size
    skip = False
    done_pdb = sampling_main(args, config, done_pdb, sampling_model, db_complex_test, batch_size, start_batch_size, logger, skip)
        


    
    
        
    