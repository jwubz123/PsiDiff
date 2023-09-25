# PsiDiff

The official implementation of paper: FROM SINGLETON IN TO PAIRWISE: GENERATING LIG- AND CONFORMATION WITH LIGAND-TARGET INTERACTIONS

## Environments
Our model runs on Tesla A100 40G GPUs, you can create the environment by conda:
```
# Clone the environment
conda env create -f environment.yml
```

## Dataset
The original PDBBind-2020 dataset can be derived at http://www.pdbbind.org.cn/index.php?action=showall
We provide our processed version of training, validation, and testing dataset at:
After downloding the dataset, it should be put into the folder path as specified in the datasets/

## Checkpoints
The trained checkpoints for both score function and energy function are saved in: 
s_theta.pt for score function, gap.pt, energy.pt and charge.pt
After downloding the checkpoints, it should be put into the folder path as specified in the logs/

## Training
All hyper-parameters and training details are provided in config files (./configs/pdbbind_default.yml), you can tune the parameters.
You can train the score function model with the following commands:
```
bash train_ddp.sh
```
You can train the energy function model with the following commands:
```
bash train_g_phi.sh
```
The model checkpoints will be saved in /logs folder

## Sampling
You can use the following commands to generate samples:
```
bash sample.sh
```
The generated mols will be saved in /samples folder
### Evaluation and visualization:
The evaluation contains RMSD and ligand RMSD. After evaluation, the ligand files will be saved in .mol format which can be opened by pymol.
To run the evaluation, run 
```
python eval.py
```
