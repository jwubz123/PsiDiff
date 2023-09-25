exp='cond_all_change_weight40_5000'
config='configs/pdbbind_default.yml' 
port=11113
NUM_NODES=1
GPUS_PER_NODE=8

srun -N $NUM_NODES --gres gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE \
python train_ddp.py --config $config --port $port --exp $exp --slurm