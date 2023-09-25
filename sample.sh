gpu_id=0
surfix=cond_all_change_weight40_5000
sigma=3.4192
index=1

    python3.8 sampling.py /samples/8_2.8_${surfix}/checkpoints/s_theta.pt \
        logs/EGNN_1000/gap_multi3/checkpoints/gap.pt \
        --guidance_path2 logs/EGNN_1000/en_multi3/checkpoints/en.pt \
        --guidance_path3 /logs/EGNN_1000/ch_multi3/checkpoints/ch.pt \
        --guidance_cond multi \
        --config_name multi\
        --start_idx 0 \
        --end_idx 337 \
        --seed ${index} \
        --gpu_id ${gpu_id} \
        --test_set datasets/equibind_data_pocket_add_std_norm_prop/test_with_charges_large.pt \
        --tag our_sample_${main_epoch}_no_no \
        --batch_size 36 \
        --n_steps 4999 \
        --global_start_sigma ${sigma}