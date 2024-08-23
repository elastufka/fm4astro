#!/bin/sh
#SBATCH --job-name finetune_GMNIST            # this is a parameter to help you sort your job when listing it
#SBATCH --error  sbatch_logs/finetune_GMNIST-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output  sbatch_logs/finetune_GMNIST-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL

export WANDB_API_KEY=api_key
export WANDB_PROJECT="supervised_GMNIST"

#ViTB/16
srun python  finetune.py --epochs 100 --output_dir "GMNIST/ViTB/10p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 800 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "GMNIST/ViTB/30p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 2400 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "GMNIST/ViTB/50p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 4000 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "GMNIST/ViTB/100p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --lr 0.000005

#RN50
srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn50/10p" --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 800

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn50/30p" --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 2400

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn50/50p" --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 4000

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn50/100p" --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" 

#RN18
srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn18/10p" --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 800

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn18/30p" --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 2400

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn18/50p" --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 4000

srun python  finetune.py --epochs 100 --output_dir "GMNIST/rn18/100p" --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15