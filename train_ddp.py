import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import pandas as pd

import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.transforms import BatchDownSamplingIndex
from utils.complex_graph import ComplexDataset

from models.s_theta_net import get_model
# from utils.datasets import ConformationDataset
from utils.misc import *
from utils.common import get_optimizer, get_scheduler
from utils.dist_util import setup_dist
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pdbbind_default.yml')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--project', type=str, default='default')
    parser.add_argument('--exp', type=str, default='run1')
    parser.add_argument('--port', type=int, default=None, help='give a master port to slurm for distributed training')
    parser.add_argument('--slurm', action='store_true', help='Use slurm for distributed training')
    parser.add_argument('--debug', action='store_true', help='debug mode, not sync with wandb')
    args = parser.parse_args()

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    
    # set ddp settings
    rank, local_rank, world_size, device = setup_dist(args, port=args.port, verbose=True)
    setattr(config, 'local_rank', local_rank)
    setattr(config, 'world_size', world_size)
    seed_all(config.train.seed + config.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Logging
    if dist.get_rank()==0:
        if resume:
            log_dir = args.config
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=args.exp)
            shutil.copytree('models', os.path.join(log_dir, 'models'))
            shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        valid_csv = f'{log_dir}/valid.csv'
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = get_logger('train', log_dir)
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
    logging.basicConfig(level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)

    # Datasets and loaders
    if dist.get_rank()==0: logger.info('Loading datasets...')
    db_complex_train = torch.load(config.dataset.train)
    db_complex_val = torch.load(config.dataset.val)

    pdbIDs_val = [db_complex_val[i][2] for i in range(len(db_complex_val))]
    pdbIDs_train = [db_complex_train[i][2] for i in range(len(db_complex_train))]
    if dist.get_rank()==0: logger.info('Complexes in training set: %d' % len(pdbIDs_train))
    if dist.get_rank()==0: logger.info('Complexes in valid set: %d' % len(pdbIDs_val))
    batch_vars = ["gen_xyz_p", "atom_coords_p", 'complex_pos', 'target_idx']
    transform = BatchDownSamplingIndex()
    dist_sampler = DistributedSampler(
        dataset=db_complex_train, num_replicas=world_size, rank=dist.get_rank()
    )
    train_loader = DataLoader(
        db_complex_train, batch_size=config.train.batch_size, follow_batch=batch_vars, 
        shuffle=False, sampler=dist_sampler, num_workers=2
    )
    val_loader = DataLoader(
        db_complex_val, batch_size=config.train.batch_size, follow_batch=batch_vars, shuffle=False
    )
    # Model
    if dist.get_rank()==0: logger.info('Building model...')
    model = get_model(config.model).to(device)
    
    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    start_epoch = 1

    # Resume from checkpoint
    if resume:
        try:
            ckpt_path, start_epoch = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
            start_epoch += 1
            if dist.get_rank()==0:
                logger.info('Resuming from: %s' % ckpt_path)
                logger.info('Epochs: %d' % start_epoch)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer_global.load_state_dict(ckpt['optimizer_global'])
            optimizer_local.load_state_dict(ckpt['optimizer_local'])
            scheduler_global.load_state_dict(ckpt['scheduler_global'])
            scheduler_local.load_state_dict(ckpt['scheduler_local'])
        except:
            print('Start from 1')
    
    # setup ddp model
    model = DDP(model, device_ids=[config.local_rank], broadcast_buffers=False)

    def train(epoch):
        model.module.train()
        train_loader.sampler.set_epoch(epoch-1) # `epoch` starts from 1
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        if epoch > config.train.change_loss_weight_epoch:
            change_loss_weight = True
        else:
            change_loss_weight = False
        for i, batch in enumerate(tqdm(train_loader)):
            batch = transform(batch)
            ligand, target, pdbid = batch
            ligand, target = ligand.to(device), target.to(device)
            loss, loss_global, loss_local = model.module.get_loss(
                ligand, 
                target,
                num_graphs=config.train.batch_size,
                anneal_power=config.train.anneal_power,
                return_unreduced_loss=True,
                change_loss_weight=change_loss_weight
            )
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)
            loss = loss.mean()

            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            loss.backward()

            orig_grad_norm = clip_grad_norm_(model.module.parameters(), config.train.max_grad_norm)
            optimizer_global.step()
            optimizer_local.step()
            
            sum_loss_global += loss_global.sum().item()
            sum_n_global += loss_global.size(0)
            sum_loss_local += loss_local.sum().item()
            sum_n_local += loss_local.size(0)
            
            # log
            if i>0 and i%config.train.log_freq==0:
                if dist.get_rank()==0:
                    avg_loss = sum_loss / sum_n
                    avg_loss_global = sum_loss_global / sum_n_global
                    avg_loss_local = sum_loss_local / sum_n_local
                    logger.info('[Train] Epoch %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f' % (
                        epoch, avg_loss, avg_loss_global, avg_loss_local, orig_grad_norm, optimizer_global.param_groups[0]['lr'], optimizer_local.param_groups[0]['lr'],
                    ))
                    stats = {
                        'train/loss': avg_loss,
                        'train/loss_global': avg_loss_global,
                        'train/loss_local': avg_loss_local,
                        'train/lr_global': optimizer_global.param_groups[0]['lr'],
                        'train/lr_local': optimizer_local.param_groups[0]['lr'],
                        'train/grad_norm': orig_grad_norm,
                        
                    }
                    wandb.log(stats, step=i+(epoch-1)*len(train_loader))
                
                sum_loss, sum_n = 0, 0
                sum_loss_global, sum_n_global = 0, 0
                sum_loss_local, sum_n_local = 0, 0
        
        return

    def validate(epoch):
        model.module.eval()
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        if epoch > config.train.change_loss_weight_epoch:
            change_loss_weight = True
        else:
            change_loss_weight = False
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = transform(batch)
                ligand, target, pdbid = batch
                ligand, target = ligand.to(device), target.to(device)
                loss, loss_global, loss_local = model.module.get_loss(
                    ligand, 
                    target,
                    num_graphs=config.train.batch_size,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True,
                    change_loss_weight=change_loss_weight
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.size(0)
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
            avg_loss = sum_loss / sum_n
            avg_loss_global = sum_loss_global / sum_n_global
            avg_loss_local = sum_loss_local / sum_n_local
            
            if config.train.scheduler.type == 'plateau':
                scheduler_global.step(avg_loss_global)
                scheduler_local.step(avg_loss_local)
            else:
                scheduler_global.step()
                scheduler_local.step()

            logger.info('[Validate] Epoch %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f' % (
                epoch, avg_loss, avg_loss_global, avg_loss_local,
            ))
            
            return avg_loss, avg_loss_global, avg_loss_local 
    try:
        for epoch in range(start_epoch, config.train.epochs + 1):
            if dist.get_rank()==0:
                wandb.log({'epoch': epoch})
            train(epoch)
            if dist.get_rank()==0:
                if epoch % config.train.save_freq == 0 or epoch == config.train.epochs:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
                    torch.save({
                        'config': config,
                        'model': model.module.state_dict(),
                        'optimizer_global': optimizer_global.state_dict(),
                        'scheduler_global': scheduler_global.state_dict(),
                        'optimizer_local': optimizer_local.state_dict(),
                        'scheduler_local': scheduler_local.state_dict(),
                        'epoch': epoch,
                    }, ckpt_path)
                    logger.info(f'{epoch}, saved!')
                if epoch % config.train.val_freq == 0 or epoch == config.train.epochs:
                    try:
                        avg_loss, avg_loss_global, avg_loss_local = validate(epoch)
                        stats = {
                            'valid/loss': avg_loss,
                            'valid/loss_global': avg_loss_global,
                            'valid/loss_local': avg_loss_local,
                        }
                        if os.path.exists(valid_csv):
                            df = pd.read_csv(valid_csv, index_col=0)
                            stats_temp = stats
                            stats_temp['epoch'] = epoch
                            df = df.append(stats_temp, ignore_index=True)
                        else:
                            df = stats
                            df['epoch'] = epoch
                            df = pd.DataFrame(df, index=[0])
                        df.to_csv(valid_csv)
                        wandb.log(stats, step=epoch)
                    except:
                        print('Validation cuda out of memory!')
                        continue
    except KeyboardInterrupt:
        logger.info('Terminating...')
        wandb.finish()

