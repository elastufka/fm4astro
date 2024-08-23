#!/bin/sh
#SBATCH --job-name mgcls_frcnn_mae            # this is a parameter to help you sort your job when listing it
#SBATCH --error sbatch_logs/mgcls_frcnn_mae-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output sbatch_logs/mgcls_frcnn_mae-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
#SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL

export WANDB_API_KEY=api_key
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29500'

#frozen, 10% labels
data=MGCLS_data/rcnn_dataset_10p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_10p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth --freeze 12 

#frozen, 30% labels
data=MGCLS_data/rcnn_dataset_30p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_30p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth --freeze 12 

#frozen, 50% labels
data=MGCLS_data/rcnn_dataset_50p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_50p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth --freeze 12 

#frozen, 100% labels
data=MGCLS_data/rcnn_dataset.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_frozen_100p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth --freeze 12 

#finetune, 10% labels
data=MGCLS_data/rcnn_dataset_10p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_finetune_10p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth 

#finetune, 30% labels
data=MGCLS_data/rcnn_dataset_30p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_finetune_30p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth 

#finetune, 50% labels
data=MGCLS_data/rcnn_dataset_50p.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_finetune_50p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth #--freeze 12 --resume

#finetune, 100% labels
data=MGCLS_data/rcnn_dataset.yaml
srun python fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_vitdet --batch 8 -ims $imsize -uta --name mgcls_vit_det_mae_finetune_100p --mosaic 0 --weights facebook_vit_mae_base_statedict.pth 