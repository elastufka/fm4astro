#!/bin/sh
#SBATCH --job-name mgcls_frcnn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/mgcls_frcnn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/mgcls_frcnn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu007,gpu011,gpu013 #,gpu006,

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml  --epochs 100 --model fasterrcnn_resnet50_fpn --name mgcls_resnet50_fpn_scratch --batch 16 -ims 512 -uta --mosaic 0  #--lr 0.00005 --freeze 8 #-dw #--freeze 8 #--weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_teacher.pth

#MSN
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml --epochs 100 --model fasterrcnn_vitdet --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth -ims 512 -uta --name mgcls_vit_det_msn512_coarse --freeze 12 --mosaic 0  #-dw -r #FM_compare/facebook_vit_msn_base_statedict.pth

#MAE
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml  --epochs 100 --model fasterrcnn_vitdet --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict.pth --freeze 12 -ims 256 -uta --name mgcls_vit_det_mae #-dw -r #

#RN50
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml  --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16  -ims 256 -uta --name mgcls_resnet50_fpn -r --weights /home/users/l/lastufka/outputs/training/mgcls_resnet50_fpn/last_model.pth #-dw -r #--weights /home/users/l/lastufka/FM_compare/facebook_vit_mae_base_statedict.pth --freeze 12

#DINOv1
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml  --epochs 120 --model fasterrcnn_vitdet --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_dinov1_base_statedict.pth --freeze 12 -ims 256 -uta --name mgcls_vit_det_dino --mosaic 0

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data /home/users/l/lastufka/scratch/rgz/od/rcnn_dataset.yaml  --epochs 100 --model fasterrcnn_vitdet --name vit_det_msn_f6 --batch 16 --weights /home/users/l/lastufka/FM_compare/facebook_vit_msn_base_statedict.pth --freeze 6 -ims 256 -uta

