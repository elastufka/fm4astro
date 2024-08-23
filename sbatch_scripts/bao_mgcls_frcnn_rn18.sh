#!/bin/sh
#SBATCH --job-name rgz_frcnn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/rgz_frcnn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/rgz_frcnn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
#SBATCH --gres=gpu:2,VramPerGpu:25G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu026,gpu044,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

imsize=512
# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_10p.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name mgcls_resnet18_finetune_10p --batch 16 -ims $imsize -uta --mosaic 0 #--freeze 8 #byol//byol_rcnn.pth

# data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_5k.yaml

# srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name mgcls_resnet18_finetune_30p --batch 16 -ims $imsize -uta --mosaic 0 #--freeze 8

data=/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs/rcnn_dataset_50p.yaml

srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name mgcls_resnet18_finetune_50p --batch 16 -ims $imsize -uta --mosaic 0 --resume --weights /home/users/l/lastufka/outputs/training/mgcls_resnet18_finetune_50p/last_model.pth -dw

#finetune
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --name mgcls_resnet18_rgz_finetune --batch 16 --weights /home/users/l/lastufka/outputs/training/mgcls_resnet18_rgz_finetune/last_model.pth -ims $imsize -uta --mosaic 0 

#from scratch
#run python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --batch 16 -ims $imsize --name mgcls_resnet18_scratch --mosaic 0 

#from pre-train - change from_pretrained = true in code
#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --batch 16 -ims $imsize --name mgcls_resnet18 --mosaic 0 

#srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --epochs 100 --model fasterrcnn_resnet18 --batch 16 -ims $imsize --name mgcls_resnet18_frozen --mosaic 0 --freeze 8