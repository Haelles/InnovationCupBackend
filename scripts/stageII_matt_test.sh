#!/bin/sh

if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0  python testII.py --name fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode fashionE  \
    --dataroot ~/data/datasets/fashionE \
    --dataroot_mask ~/data/datasets/mask/irr/mask/testing_mask_dataset \
    --dataroot_color_mask ~/data/datasets/mask/color/qd_imd \
    --output_nc 3 --input_nc 29 --stage 25 --model stageII_multiatt3 --netG multiatt3 --norm_G batch \
    --gpu_ids 0 --display_winsize 960 --ngf 64  --input_ANLs_nc 5 --which_epoch 20


elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=2  python testII.py --name deepfashion_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode deepfashion  \
    --dataroot /data/datasets/deepfashion \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --output_nc 3 --input_nc 29 --stage 25 --model stageII_multiatt3 --netG multiatt3 --norm_G batch \
    --gpu_ids 0 --display_winsize 960 --ngf 64  --input_ANLs_nc 5 --which_epoch 50


elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=2  python testII.py --name mpv_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode mpv  \
    --dataroot /data/datasets/MPV \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --output_nc 3 --input_nc 29 --stage 25 --model stageII_multiatt3 --netG multiatt3 --norm_G batch \
    --gpu_ids 0 --display_winsize 960 --ngf 64  --input_ANLs_nc 5 --which_epoch 50



fi


