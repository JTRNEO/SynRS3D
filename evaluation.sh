#!/bin/bash
python evaluation.py \
--restore_path /path/of/your/model \
--num_classes 8 \
--test_datasets DFC18 DFC19_JAX  \
--ood_datasets DFC19_JAX \
--multi_task \
--images_file train.txt test.txt test.txt \
--save_num_images 0 \
--snapshot_dir /result