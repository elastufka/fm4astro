#!/bin/sh
#SBATCH --job-name rgz_frcnn_rn            # this is a parameter to help you sort your job when listing it
#SBATCH --error sbatch_logs/rgz_frcnn_rn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output sbatch_logs/rgz_frcnn_rn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
#SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu004,gpu026

export WANDB_API_KEY=api_key
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

imsize=256

data=rgz/rcnn_dataset_10p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_frozen_10p --mosaic 0 --freeze 8

data=rgz/rcnn_dataset_30p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_frozen_30p --mosaic 0 --freeze 8

data=rgz/rcnn_dataset_50p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_frozen_50p --mosaic 0 --freeze 8

data=rgz/rcnn_dataset.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_frozen_100p --mosaic 0 --freeze 8

data=rgz/rcnn_dataset_10p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_finetune_10p --mosaic 0 

data=rgz/rcnn_dataset_30p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_finetune_30p --mosaic 0 

data=rgz/rcnn_dataset_50p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_finetune_50p --mosaic 0 

data=rgz/rcnn_dataset.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 8 -ims $imsize -uta --name rgz_resnet50_finetune_100p --mosaic 0