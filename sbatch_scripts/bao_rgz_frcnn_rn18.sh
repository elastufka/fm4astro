#!/bin/sh
#SBATCH --job-name rgz_frcnn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/rgz_frcnn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/rgz_frcnn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu005 #,gpu006,gpu007,gpu011

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

#data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_10p.yaml
imsize=256

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18_tf --name resnet18_tf_smarthulk_frozen_lr02 --batch 16 -uta --mosaic 0 -ims $imsize --val_freq 5 -ca --lr 0.002 --weights /home/users/l/lastufka/riggi-SimCLR/encoder_weights-resnet18_simclr_smarthulk256-smgps_ch3_500epochs_msd.pth --freeze 8 #--lr 0.0005  #hayat_gz/mocov2_encoder_k_frcnn.pth hayat_gz/mocov2_encoder_k_frcnn.pth #

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_finetune_10p --batch 16 -uta --mosaic 0 -ims $imsize 

data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_30p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_finetune_30p --batch 16 -uta --mosaic 0 -ims $imsize --resume --weights /home/users/l/lastufka/outputs/training/resnet18_finetune_30p/last_model.pth 

data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_50p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_finetune_50p --batch 16 -uta --mosaic 0 -ims $imsize 


data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_10p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_frozen_10p --batch 16 -uta --mosaic 0 -ims $imsize --freeze 8

data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_30p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_frozen_30p --batch 16 -uta --mosaic 0 -ims $imsize  --freeze 8

data=/home/users/l/lastufka/scratch/rgz/od/rcnn_dataset_50p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name resnet18_frozen_50p --batch 16 -uta --mosaic 0 -ims $imsize  --freeze 8
