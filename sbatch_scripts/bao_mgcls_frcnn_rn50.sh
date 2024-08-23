#!/bin/sh
#SBATCH --job-name mgcls_frcnn_rn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_rn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/mgcls_frcnn_rn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu017,gpu046,gpu047
,
export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

imsize=512

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml
# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16 -ims $imsize --name mgcls_resnet50_finetune_10p --mosaic 0  #--freeze 8 #--weights /home/users/l/lastufka/outputs/training/mgcls_resnet50_mgcls_frozen/last_model.pth --lr 0.0001 --resume #--in_chans 1 #--lr 0.0001
#hayat_gz/mocov2_encoder_k_frcnn.pth DINO/mgcls_resnet50_pretrain_425b_teacher.pth

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml
# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16 -ims $imsize --name mgcls_resnet50_finetune_30p --mosaic 0 --resume --weights /home/users/l/lastufka/outputs/training/mgcls_resnet50_finetune_30p/last_model.pth -dw #--freeze 8

data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_50p.yaml
srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16 -ims $imsize --name mgcls_resnet50_finetune_50p --mosaic 0  --weights /home/users/l/lastufka/outputs/training/mgcls_resnet50_finetune_50p/last_model.pth #-dw # --freeze 8

#finetune
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16 -ims $imsize --name mgcls_resnet50_mgcls_finetune --mosaic 0 --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_teacher.pth 

#1chan
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn_1chan --batch 16 -ims $imsize --name mgcls_resnet50_mgcls350b_frozen_10p --mosaic 0  --freeze 8 --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_350b_teacher_1chan.pth -dw --in_chans 1 #--lr 0.0001
#hayat_gz/mocov2_encoder_k_frcnn.pth 
#-r --weights /home/users/l/lastufka/outputs/training/mgcls_resnet50_fpn512_frozen_50p/last_model.pth

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet50_fpn --batch 16 -ims $imsize -uta --name mgcls_resnet50_gz2_finetune_30p --mosaic 0  --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_frcnn.pth  -dw # --freeze 8
