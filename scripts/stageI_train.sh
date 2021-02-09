#!/bin/sh


if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0  python trainI.py --name fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode fashionE \
    --dataroot ~/data/datasets/fashionE \
    --dataroot_mask ~/data/datasets/mask/irr/mask/testing_mask_dataset \
    --dataroot_color_mask ~/data/datasets/mask/color/qd_imd \
    --batchSize 5 --gpu_ids 0 --tf_log --output_nc 20 --input_nc 26 --stage 1 --no_vgg_loss --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch --lambda_L1 10 --lambda_feat 10 --lambda_parsing 10 --lambda_gan 1


elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=3,2,1,0  python trainI.py --name deepfashion_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode deepfashion \
    --dataroot /data/datasets/deepfashion \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --batchSize 20 --gpu_ids 0,1,2,3 --tf_log --output_nc 20 --input_nc 26 --stage 1 --no_vgg_loss --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch --lambda_L1 10 --lambda_feat 10 --lambda_parsing 10 --lambda_gan 1


elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=3,2,1,0  python trainI.py --name mpv_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520 --dataset_mode mpv \
    --dataroot /data/datasets/MPV \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --batchSize 20 --gpu_ids 0,1,2,3 --tf_log --output_nc 20 --input_nc 26 --stage 1 --no_vgg_loss --no_flip \
    --model stageI_parsing --netG parsing --norm_G batch --lambda_L1 10 --lambda_feat 10 --lambda_parsing 10 --lambda_gan 1


elif [ $1 == 0 ]; then
    CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir=fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520:./checkpoints/fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520/logs,\
deepfashion_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520:./checkpoints/deepfashion_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520/logs,\
mpv_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520:./checkpoints/mpv_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520/logs --port 5588

fi

