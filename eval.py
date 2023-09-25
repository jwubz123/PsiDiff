import os
import pickle
import numpy as np
import pandas as pd
import argparse
from utils.misc import *
from utils.save_mol import save_mol
import glob
from pathlib import Path
import statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='samples/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--cal_ff', action='store_true', default=False)
    args = parser.parse_args()
    assert os.path.isdir(args.path)

    # Logging
    tag = args.path.split('/')[-1].split('.')[0]
    logger = get_logger('eval', os.path.dirname(args.path), 'log_eval_'+tag+'.txt')
    real_root = '/shared_space/xtalpi_lab/Datasets/PDBbind/PDBbind_v2020/v2020_all'
    save_root = f'{args.path}'
    Path(save_root).mkdir(exist_ok=True)  
    Path(f'{save_root}/ff_aligned_mol').mkdir(exist_ok=True)  
    Path(f'{save_root}/ff_mol').mkdir(exist_ok=True)  
    Path(f'{save_root}/gen_mol').mkdir(exist_ok=True) 
    Path(f'{save_root}/aligned_gen_mol').mkdir(exist_ok=True)   
    # Load results
    logger.info('Loading results: %s' % args.path)
    results = {}
    csv_fn = args.path + '/rmsd.csv'
    
    best_rmsd_list = []
    rmsd_list = []
    centroid_dist_list = []

    ff_best_rmsd_list = []
    ff_rmsd_list = []
    ff_centroid_dist_list = []

    pdb_list = []
    for p in glob.glob(f'{args.path}/samples_*.pkl'):

        pdb_id = Path(p).stem[-4:]
        with open(f'{args.path}/samples_{pdb_id}.pkl', 'rb') as f:
            ligand = pickle.load(f)
            f.close()

        # Evaluator
        # best_rmsd, rmsd, ff_best_rmsd, ff_rmsd = save_mol(ligand, pdb_id, real_root, save_root=save_root)
        try:
            best_rmsd, rmsd, centroid_dist, ff_best_rmsd, ff_rmsd, ff_centroid_dist = save_mol(ligand, pdb_id, real_root, save_root=save_root, try_num=0, cal_ff=args.cal_ff)
        except:
            best_rmsd, rmsd, centroid_dist, ff_best_rmsd, ff_rmsd, ff_centroid_dist = save_mol(ligand, pdb_id, real_root, save_root=save_root, try_num=1, cal_ff=args.cal_ff)

        rmsd_list.append(rmsd.cpu().item())
        best_rmsd_list.append(best_rmsd)
        centroid_dist_list.append(centroid_dist.cpu().item())
        
        ff_best_rmsd_list.append(ff_best_rmsd)
        ff_rmsd_list.append(ff_rmsd.cpu().item())
        ff_centroid_dist_list.append(ff_centroid_dist.cpu().item())
        
        pdb_list.append(pdb_id)

    results['pdb_id'] = pdb_list

    results['best_rmsd'] = best_rmsd_list
    results['rmsd'] = rmsd_list
    results['centroid_dist'] = centroid_dist_list

    results['ff_best_rmsd'] = ff_best_rmsd_list
    results['ff_rmsd'] = ff_rmsd_list
    results['ff_centroid_dist'] = ff_centroid_dist_list
    
    # # Save results
    
    pd.DataFrame(results).to_csv(csv_fn)
    # logger.info(f'Mean and Median best_RMSD for {len(pdb_list)} ligands: {sum(best_rmsd_list)/len(best_rmsd_list)}, {statistics.median(best_rmsd_list)}')
    # logger.info(f'Mean and Median ff_best_RMSD for {len(pdb_list)} ligands: {sum(ff_best_rmsd_list)/len(ff_best_rmsd_list)}, {statistics.median(ff_best_rmsd_list)}')
    # logger.info(f'Mean and Median RMSD for {len(pdb_list)} ligands: {sum(rmsd_list)/len(rmsd_list)}, {statistics.median(rmsd_list)}')
    # logger.info(f'Mean and Median ff_RMSD for {len(pdb_list)} ligands: {sum(ff_rmsd_list)/len(ff_rmsd_list)}, {statistics.median(ff_rmsd_list)}')
    # logger.info(f'Mean and Median centroid_dist for {len(pdb_list)} ligands: {sum(centroid_dist_list)/len(centroid_dist_list)}, {statistics.median(centroid_dist_list)}')
    # logger.info(f'Mean and Median ff_centroid_dist for {len(pdb_list)} ligands: {sum(ff_centroid_dist_list)/len(ff_centroid_dist_list)}, {statistics.median(ff_centroid_dist_list)}')
    
    stat_list = ['best_rmsd', 'ff_best_rmsd', 'rmsd', 'ff_rmsd', 'centroid_dist']
    for stat in stat_list:

        eval_pd = pd.read_csv(csv_fn)
        align_rmsd_summary = {}
        align_rmsd = eval_pd[stat]
        below_2 = align_rmsd[align_rmsd <= 2]
        below_5 = align_rmsd[align_rmsd <= 5]
        below_half = align_rmsd[align_rmsd <= 0.5]
        align_rmsd_summary['mean'] = [align_rmsd.mean()]
        align_rmsd_summary['q25%'] = [align_rmsd.quantile(q=0.25)]
        align_rmsd_summary['q50%'] = [align_rmsd.quantile(q=0.5)]
        align_rmsd_summary['q75%'] = [align_rmsd.quantile(q=0.75)]
        align_rmsd_summary['<= 0.5A'] = [len(below_half) / len(align_rmsd)]
        align_rmsd_summary['<= 2A'] = [len(below_2) / len(align_rmsd)]
        align_rmsd_summary['<= 5A'] = [len(below_5) / len(align_rmsd)]
        logger.info(f'{len(align_rmsd)} results: {stat}')
        logger.info(pd.DataFrame.from_dict(align_rmsd_summary))
        logger.info('---------------------------------------------')
        if stat == 'best_rmsd':
            mean_aligned = align_rmsd_summary['mean']
            median_aligned = align_rmsd_summary['q50%']
        if stat == 'rmsd':
            mean_rmsd = align_rmsd_summary['mean']
            median_rmsd = align_rmsd_summary['q50%']
    logger.info('RMSD results saved as: %s' % csv_fn)
    print('Aligned: {:.4f}  {:.4f}'.format(mean_aligned[0], median_aligned[0]))
    print('RMSD: {:.4f}  {:.4f}'.format(mean_rmsd[0], median_rmsd[0]))