#!/bin/sh
#SBATCH --job-name mgcls_frcnn_msn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_msn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_msn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
#SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu017,gpu046,gpu047,gpu026

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29500'

data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml
imsize=512

#MSN
# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_msn_finetune_10p --mosaic 0  --lr 0.0005 --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth #--freeze 12 #/home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict_tv.pth  #--freeze 12  #/home/users/l/lastufka/segment-anything/SAM_pretrained_state_dict_tv.pth #--resume  #/home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth  --weights /home/users/l/lastufka/outputs/training/mgcls_vit_det_msn512_frozen_50p/last_model.pt

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_msn_frozen_30p --mosaic 0  --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth --resume --weights /home/users/l/lastufka/outputs/training/mgcls_vit_det_msn_frozen_30p/last_model.pth --freeze 12

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_50p.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_msn_frozen_50p --mosaic 0  --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth --freeze 12

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml
# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_10p --mosaic 0  --weights /home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict.pth --freeze 12 #/home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict_tv.pth  #--freeze 12  #/home/users/l/lastufka/segment-anything/SAM_pretrained_state_dict_tv.pth #--resume  #/home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth  --weights /home/users/l/lastufka/outputs/training/mgcls_vit_det_msn512_frozen_50p/last_model.pt --lr 0.0005 

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_30p --mosaic 0 --weights /home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict.pth --freeze 12

data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_50p.yaml

srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_msn_frozen_50p --mosaic 0 --weights /home/users/l/lastufka/outputs/training/mgcls_vit_det_msn_frozen_50p/last_model.pth --resume --freeze 12 #/home/users/l/lastufka/outputs/training/mgcls_vit_det_msn_finetune_50p --resume #--freeze 12