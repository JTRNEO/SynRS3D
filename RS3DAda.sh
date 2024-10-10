#!/bin/bash
train_set=('grid_g05_mid_v2' 'grid_g005_mid_v2' 'terrain_g05_low_v1' 'terrain_g005_low_v1' 'grid_g05_mid_v1' 'grid_g005_mid_v1' 'terrain_g05_mid_v1' 'terrain_g005_mid_v1' 'grid_g05_low_v1' 'grid_g005_low_v1' 'grid_g05_high_v1' 'grid_g005_high_v1' 'terrain_g05_high_v1' 'terrain_g005_high_v1' 'terrain_g1_low_v1' 'terrain_g1_mid_v1' 'terrain_g1_high_v1')
test_set=('DFC18' 'DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATL')

images_file=('train.txt' 'test_syn.txt' 'train.txt')
da=('FDA' 'HM' 'PDA')

/home/songjian/anaconda3/envs/mmseg/bin/python train_dpt_RS3DAda.py \
--datasets ${train_set[*]} \
--test_datasets ${test_set[*]} \
--ood_datasets ${test_set[*]} \
--pesudo_datasets ${test_set[*]} \
--crop_size 392 \
--encoder vitl \
--decoder DPT \
--apply_da ${da[*]} \
--FDA_beta 0.01 \
--HM_blend_ratio 0.8 1.0 \
--PDA_blend_ratio 0.8 1.0 \
--PDA_type standard \
--tgt_datasets ${test_set[*]} \
--snapshot_dir /path/to/your/project/SynRS3D/snapshot_rs3dada \
--images_file ${images_file[*]} \
--batch_size 1 \
--learning_rate 1e-6 \
--weight_decay 5e-4 \
--warmup_steps 1500 \
--warmup_mode linear \
--decay_mode poly \
--num_steps 40000 \
--save_num_images 0 \
--save_pred_every 500 \
--multi_task \
--pretrained \
--shuffle \
--only_save_best \
--feat_loss \
--fl_threshold 0.80 \
--fl_start 3 \
--fl_weight 1.0 \
--fl_decrement 0 \
--pesudo_threshold 0.95 \
--mix_type 'ClassMix' \
--ema_alpha 0.99 \
--pesudo_weight_type 'he' \
--pesudo_dsm_threshold 1.55 \
--pesudo_file 'train.txt' \
--use_ground_mask \
--decoder_lr_weight 10 \
--max_da_images 200 \
--lambda_dsms 0.8
#optional
#--eval_oem