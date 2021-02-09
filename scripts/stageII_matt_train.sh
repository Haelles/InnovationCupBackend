#!/bin/sh

if [ $1 == 1 ]; then
    CUDA_VISIBLE_DEVICES=0  python trainII.py --name fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode fashionE \
    --dataroot ~/data/datasets/fashionE \
    --dataroot_mask ~/data/datasets/mask/irr/mask/testing_mask_dataset \
    --dataroot_color_mask ~/data/datasets/mask/color/qd_imd \
    --batchSize 4 --gpu_ids 0 --tf_log --no_flip --output_nc 3 --input_nc 29 --stage 25  \
    --model stageII_multiatt3 --netG multiatt3 --netD inpaint --ngf 64 --norm_G batch \
    --input_ANLs_nc 5 --no_parsing_loss --no_ganFeat_loss --no_L1_loss --lambda_style 200 --lambda_tv 10

elif [ $1 == 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3  python trainII.py --name deepfashion_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode deepfashion \
    --dataroot /data/datasets/deepfashion \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --batchSize 8 --gpu_ids 0,1,2,3 --tf_log --no_flip --output_nc 3 --input_nc 29 --stage 25  \
    --model stageII_multiatt3 --netG multiatt3 --netD inpaint --ngf 64 --norm_G batch \
    --input_ANLs_nc 5 --no_parsing_loss --no_ganFeat_loss --no_L1_loss --lambda_style 200 --lambda_tv 10

elif [ $1 == 3 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3  python trainII.py --name mpv_stageII_matt3_noFeat_noL1_tv10_style200_20190520 --dataset_mode mpv \
    --dataroot /data/datasets/MPV \
    --dataroot_mask /data/datasets/mask_and_colormask/mask/mask/testing_mask_dataset \
    --dataroot_color_mask /data/datasets/mask_and_colormask/color_mask/qd_imd \
    --batchSize 8 --gpu_ids 0,1,2,3 --tf_log --no_flip --output_nc 3 --input_nc 29 --stage 25  \
    --model stageII_multiatt3 --netG multiatt3 --netD inpaint --ngf 64 --norm_G batch \
    --input_ANLs_nc 5 --no_parsing_loss --no_ganFeat_loss --no_L1_loss --lambda_style 200 --lambda_tv 10

elif [ $1 == 0 ]; then
    CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir=fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520:./checkpoints/fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520/logs,\
deepfashion_stageII_matt3_noFeat_noL1_tv10_style200_20190520:./checkpoints/deepfashion_stageII_matt3_noFeat_noL1_tv10_style200_20190520/logs,\
mpv_stageII_matt3_noFeat_noL1_tv10_style200_20190520:./checkpoints/mpv_stageII_matt3_noFeat_noL1_tv10_style200_20190520/logs --port 5599

fi


