#!/bin/bash
train_set=('sr_05_cd_aux' 'sr_005_cd_aux' 'sp_05_cd_lower' 'sp_005_cd_lower' 'sr_05_cd' 'sr_005_cd' 'sp_05_cd' 'sp_005_cd' 'sr_05_cd_lower' 'sr_005_cd_lower' 'sr_05_cd_higher' 'sr_005_cd_higher' 'sp_05_cd_higher' 'sp_005_cd_higher' 'sp_1_cd_lower' 'sp_1_cd' 'sp_1_cd_higher')
test_set=('DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATL')

images_file=('selected_train.txt' 'test_syn.txt' 'train.txt')
da=()

python train_dpt_sourceonly.py \
--datasets ${train_set[*]} \
--test_datasets ${test_set[*]} \
--ood_datasets ${test_set[*]} \
--crop_size 392 \
--encoder vitl \
--decoder DPT \
--snapshot_dir /home/songjian/project/SynRS3D/snapshot_sourceonly_test \
--images_file ${images_file[*]} \
--batch_size 2 \
--learning_rate 1e-6 \
--weight_decay 5e-4 \
--warmup_steps 0 \
--decay_mode poly \
--num_steps 40000 \
--save_num_images 0 \
--save_pred_every 500 \
--multi_task \
--pretrained \
--shuffle \
--only_save_best \
--decoder_lr_weight 10 \
--lambda_dsms 1.0 \
--eval_oem