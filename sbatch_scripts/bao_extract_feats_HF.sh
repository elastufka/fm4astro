#!/bin/sh
#SBATCH --job-name extract_feats_HF            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/extract_feats_HF-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/extract_feats_HF-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-cpu         # the partition to use. By default debug-cpu
##SBATCH --gpus 1
#SBATCH --mem=20G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 05:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu035

pip install transformers
export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92

#srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --train false

#mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_testfeat_dinov2B.pth

#srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest 

#mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_trainfeat_dinov2B.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/vit-mae-base

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_trainfeat_maeB.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/vit-mae-base --test

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_testfeat_maeB.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/vit-msn-base

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_trainfeat_msnB.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/vit-msn-base --test

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_testfeat_msnB.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/detr-resnet-50

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_trainfeat_dRN50.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model facebook/detr-resnet-50 --test

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_testfeat_dRN50.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model microsoft/resnet50

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_trainfeat_RN50.pth

srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train /home/users/l/lastufka/scratch/MiraBest --model microsoft/resnet50 --test

mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/MiraBest_testfeat_RN50.pth