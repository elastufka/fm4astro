#!/bin/sh
#SBATCH --job-name rgz_frcnn_dino            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/rgz_frcnn_dino-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/rgz_frcnn_dino-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu005,gpu007 #,gpu036 #,gpu004,gpu006,gpu007,gpu009,gpu014,gpu024,gpu046,gpu021,gpu038 #,gpu034,gpu046,gpu044


export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

#data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_10p.yaml
imsize=256

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --name vit_det_mgcls_frozen --batch 16 -uta --mosaic 0 --weights /home/users/l/lastufka/outputs/training/vit_det_mgcls_frozen/last_model.pth -ims $imsize --freeze 12 --resume #FM_compare/facebook_dinov1_base_statedict.pth

#finetune
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --name vit_det_dino_finetune_10p --batch 16 -uta --mosaic 0 --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth -ims $imsize #--freeze 12 #FM_compare/facebook_dinov1_base_statedict.pth DINO/mgcls_vitbase16_finetune_teacher_025.pth

#data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_30p.yaml
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --name vit_det_dino_finetune_30p --batch 16 -uta --mosaic 0 --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth -ims $imsize 

data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_50p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --name vit_det_dino_finetune_50p --batch 16 -uta --mosaic 0 --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth -ims $imsize 