#!/bin/sh

if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0  python testI.py --name fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode fashionE \
    --dataroot ~/data/datasets/fashionE \
    --dataroot_mask ~/data/datasets/mask/irr/mask/testing_mask_dataset \
    --dataroot_color_mask ~/data/datasets/mask/color/qd_imd \
    --gpu_ids 0 --output_nc 20 --input_nc 26 --stage 1 --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch  --which_epoch 20


elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=2  python testII.py --name deepfashion_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode deepfashion \
    --dataroot /data/datasets/deepfashion \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --gpu_ids 0 --output_nc 20 --input_nc 26 --stage 1 --no_vgg_loss --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch  --which_epoch 50


elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=2  python testII.py --name mpv_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode deepfashion \
    --dataroot /data/datasets/MPV \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --gpu_ids 0 --output_nc 20 --input_nc 26 --stage 1 --no_vgg_loss --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch  --which_epoch 50


fi


