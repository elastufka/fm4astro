#!/bin/sh
#SBATCH --job-name rgz_frcnn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/rgz_frcnn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/rgz_frcnn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu020,gpu014 #21,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml --epochs 200 --model fasterrcnn_resnet18 --name resnet18_finetune --batch 16 -ims 256 -uta --mosaic 0 --resume --weights /home/users/l/lastufka/outputs/training/resnet18_finetune/last_model.pth #--freeze 8

#10207583 #maybe the bad MAE is because it wasn't fozen so much before?
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml  --epochs 100 --model fasterrcnn_vitdet --name vit_det_mae --batch 16 --weights /home/users/l/lastufka/outputs/training/vit_det_mae/last_model.pth --freeze 12 -ims 256 -uta -dw -r #/home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict.pth

#MSN
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml --epochs 100 --model fasterrcnn_vitdet --name vit_det_msn_frozen_grid --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth -ims 256 -uta --mosaic 0 --freeze 12

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml --epochs 100 --model fasterrcnn_vitdet_14 --name vit_det_dino_f6 --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_dinov2_base_statedict_noembed.pth --freeze 6 -ims 256 -uta

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml  --epochs 100 --model fasterrcnn_resnet152 --name resnet152 --batch 16 -uta --mosaic 0 --freeze 7 --weights /home/users/l/lastufka/FM_compare/microsoft_resnet152_statedict.pth -ims 256
