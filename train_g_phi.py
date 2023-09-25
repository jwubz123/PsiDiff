
import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob

import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from torch_geometric.utils import to_dense_batch

from utils.transforms import BatchDownSamplingIndex
from utils.complex_graph import ComplexDataset
from utils.misc import *
from utils.energy_utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask,check_mask_correct
import utils.energy_utils as energy_utils
from models.g_phi_net.models import EGNN_energy
from models.g_phi_net.en_diffusion import EnergyDiffusion

def get_model(config, args, device):
    #in_node_nf: the numbder of atom type = 2 if not use atom type in the energy function
    in_node_nf = config.in_node_nf
    if config.condition_time:
        in_node_nf = in_node_nf + 1
    context_node_nf = len(config.condition_prop)
    if config.train_condition == 'atom_charges':
        out_dim = config.max_num_atoms
    else:
        out_dim = 1
    net_energy = EGNN_energy(
        in_node_nf=in_node_nf, context_node_nf=context_node_nf,
        n_dims=3, device=device, hidden_nf=config.hidden_nf,
        act_fn=torch.nn.SiLU(), n_layers=config.n_layers,
        attention=config.attention, tanh=config.tanh, mode=config.mode, norm_constant=config.norm_constant,
        inv_sublayers=config.inv_sublayers, sin_embedding=config.sin_embedding,
        normalization_factor=config.normalize_factors, aggregation_method=config.aggregation_method,
        out_dim=out_dim)
    
    guidance = EnergyDiffusion(
            dynamics=net_energy,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=config.diffusion_steps,
            noise_schedule=config.diffusion_noise_schedule,
            noise_precision=config.diffusion_noise_precision,
            norm_values=config.normalization_factor,
            include_charges=config.include_charges
        )
    return guidance

def add_cond(ligand, config, device):
    context = None
    bs, n_nodes, _ = to_dense_batch(ligand.x, ligand.batch)[0].shape

    for prop_name_str in config.condition_prop:

        if prop_name_str == 'prop_gap':
            prop = ligand.prop_gap.float().to(device=device)
            prop = prop.view(bs, 1).repeat(1, n_nodes)
            prop = prop.view(bs * n_nodes, 1)
            mean_prop = config.norm_gap[0]
            std_prop = config.norm_gap[1]
        elif prop_name_str == 'prop_energy':
            prop = ligand.prop_energy.to(device=device)
            prop = prop.view(bs, 1).repeat(1, n_nodes)
            prop = prop.view(bs * n_nodes, 1)
            mean_prop = config.norm_energy[0]
            std_prop = config.norm_energy[1]
        elif prop_name_str == 'atom_charges':
            prop = ligand.atom_charges
            prop, _ = to_dense_batch(prop, ligand.batch)
            prop = prop.view(bs * n_nodes, 1).to(device)
            mean_prop = config.norm_charge[0]
            std_prop = config.norm_charge[1]
        else: 
            raise NotImplementedError(prop_name_str)
        prop = (prop - mean_prop) / std_prop
        if context is None:
            context = prop
        else:
            context = torch.cat([context, prop], dim=1)
    return context

def cal_label(config, ligand, device):
    if config.model.energy_model.train_condition == 'prop_gap':
        properties = ligand.prop_gap.float().to(device=device)
        mean_context = config.model.energy_model.norm_gap[0]
        std_context = config.model.energy_model.norm_gap[1]
    elif config.model.energy_model.train_condition == 'prop_energy':
        properties = ligand.prop_energy.to(device=device)
        mean_context = config.model.energy_model.norm_energy[0]
        std_context = config.model.energy_model.norm_energy[1]
    elif config.model.energy_model.train_condition == 'atom_charges':
        properties = ligand.atom_charges.to(device=device)
        properties, _ = to_dense_batch(properties, ligand.batch.to(device))
        batch_size, n_prop = properties.shape
        mean_context = config.model.energy_model.norm_charge[0]
        std_context = config.model.energy_model.norm_charge[1]
    else: 
        raise NotImplementedError(config.model.energy_model.train_condition)

    label = (properties - mean_context) / std_context
    return label.to(device)

def reshape_tensors(ligand, config, device):
    h, _ = to_dense_batch(ligand.x , ligand.batch, fill_value=0.) #[B, n_node, 28]
    x, node_mask = to_dense_batch(ligand.pos, ligand.batch, fill_value=0.)
    batch_size, n_nodes, h_dim = h.size()
    if config.model.energy_model.train_condition == 'atom_charges':
        node_mask_target = torch.zeros(batch_size, config.model.energy_model.max_num_atoms).bool()
        node_mask_target[:, :n_nodes] = node_mask
        out_mask = node_mask_target.to(device)
    else:
        out_mask = None
    h, x, node_mask = h.to(device), x.to(device), node_mask.to(device)
    
    node_mask = node_mask.unsqueeze(-1)

    bs, n_nodes, n_dims = x.size()
    edge_mask = node_mask.squeeze(-1).unsqueeze(1) * node_mask

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(edge_mask.device)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    
    x = remove_mean_with_mask(x, node_mask)
    
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    assert_correctly_masked(x, node_mask)
    return h, x, node_mask, edge_mask, out_mask, n_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pdbbind_default.yml')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--project', type=str, default='energy')
    parser.add_argument('--exp', type=str, default='debug_run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true', help='debug mode, not sync with wandb')
    args = parser.parse_args()
    device = args.device
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)

    # Logging

    if resume:
        log_dir = args.config
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=args.exp)
        if args.debug is not True:
            shutil.copytree('models', os.path.join(log_dir, 'models'))
            shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    if args.debug is not True:
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = get_logger('train', log_dir)
    else:
        logger = get_logger('train', None)
        # os.rmdir(log_dir)
    logger.info(args)
    logger.info(config)
    log_name = os.path.basename(log_dir)
    # setup wandb
    import wandb
    if wandb.run is None:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project=args.project,
            name=log_name,
            mode='disabled' if args.debug is True else 'online',
        )
        wandb.config.update(args)

    # Datasets and loaders
    
    db_complex_train = torch.load(config.dataset.train)
    db_complex_val = torch.load(config.dataset.val)
    
    pdbIDs_val = [db_complex_val[i][2] for i in range(len(db_complex_val))]
    pdbIDs_train = [db_complex_train[i][2] for i in range(len(db_complex_train))]
  
    batch_vars = ["gen_xyz_p", "atom_coords_p", 'complex_pos', 'target_idx']
    transform = BatchDownSamplingIndex()
    
    train_loader = DataLoader(
        db_complex_train, batch_size=args.batch_size, follow_batch=batch_vars, 
        shuffle=True, num_workers=2
    )
    
    # Model
    
    # model = get_model(config.model).to(device)
    model = get_model(config.model.energy_model, args, device)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.n_epochs)

    gradnorm_queue = energy_utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.


    # Resume from checkpoint
    if resume:
        try:
            ckpt_path, start_epoch = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
            start_epoch = start_epoch + 1
            logger.info('Resuming from: %s' % ckpt_path)
            logger.info('Epochs: %d' % start_epoch)
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
        except:
            start_epoch = 1
            print('start from 1!')
    else:
        start_epoch = 1


    def train(epoch):
        model.train()
        sum_loss, counter = 0, 0
        loss_arr = []
        for i, batch in enumerate(tqdm(train_loader)):
            batch = transform(batch)
            ligand, target, pdbid = batch
            
            #reshape features and masks
            h, x, node_mask, edge_mask, out_mask, n_nodes = reshape_tensors(ligand, config, device)
            #obtain contexts
            context = add_cond(ligand, config.model.energy_model, device)
            #cal labels
            label = cal_label(config, ligand, device)

            optim.zero_grad()
            loss = model(
                x,
                h,
                node_mask,
                edge_mask,
                out_mask,
                context,
                label
            )
            
            loss.backward()
            
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optim.step()
            
            sum_loss += loss.item() * args.batch_size
            counter += args.batch_size
            loss_arr.append(loss.item())

            # log
            if i>0 and i % config.train.log_freq == 0:
                
                avg_loss = sum(loss_arr[-10:]) / len(loss_arr[-10:])
                
                logger.info('[Train] Iter %05d | Loss %.5f' % (
                    i, avg_loss
                ))
                stats = {'loss': avg_loss, 'grad_norm': orig_grad_norm}
                wandb.log(stats, step=i+(epoch-1)*len(train_loader))
        
        return sum_loss / counter

    
    # try:
    min_loss = 10000
    print('start_epoch: ', start_epoch)
    for epoch in range(start_epoch, args.n_epochs + 1):
        
        wandb.log({'epoch': epoch})
        train_loss = train(epoch)

        logger.info('[Train] Epoch %05d | Loss %.5f' % (
                    epoch, train_loss
                ))

        
        if epoch % config.train.save_freq == 0 or epoch == args.n_epochs:
            if args.debug is not True:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, ckpt_path)

    