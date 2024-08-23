#!/bin/sh
#SBATCH --job-name mgcls_frcnn_dino            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_dino-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_dino-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
#SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu004,gpu026

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

imsize=512

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml  --epochs 100 --model fasterrcnn_resnet50_fpn --name mgcls_resnet50_dino_10p_frozen --batch 16 -ims $imsize -uta --mosaic 0 --weights /home/users/l/lastufka/DINO/dino_resnet50_pretrain.pth --lr 0.00001 --freeze 8 --unfreeze_first #-dw


#MSN
# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_dino_finetune_10p --mosaic 0  -dw --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth #--freeze 12 #/home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025.pth #outputs/training/mgcls_vit_det_mgcls_frozen_30p/last_model.pth -r #DINO/mgcls_vitbase16_finetune_teacher_025.pth #-r #/home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth /home/users/l/lastufka/outputs/training/mgcls_vit_det_dino512_frozen_50p/last_model.pth  --weights /home/users/l/lastufka/DINO/mgcls_vitbase8_finetune_teacher_025.pth  --freeze 12

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_dino_finetune_30p --mosaic 0 --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth --freeze 12

data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_50p.yaml

srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_dino_finetune_50p --mosaic 0 --resume --weights /home/users/l/lastufka/outputs/training/mgcls_vit_det_dino_finetune_50p/last_model.pth 