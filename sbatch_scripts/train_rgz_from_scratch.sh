#!/bin/sh
#SBATCH --job-name finetune_RGZ            # this is a parameter to help you sort your job when listing it
#SBATCH --error  sbatch_logs/finetune_RGZ-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output  sbatch_logs/finetune_RGZ-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL

export WANDB_API_KEY=api_key
export WANDB_PROJECT="supervised_RGZ"

data_path=/path/to/rgz/train_single/images

#ViTB/16
srun python  finetune.py --epochs 100 --output_dir "RGZ/ViTB/10p" --dataset_train $data_path --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 469 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "RGZ/ViTB/30p" --dataset_train $data_path --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 1408 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "RGZ/ViTB/50p" --dataset_train $data_path --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --nlabels 2346 --lr 0.000005

srun python  finetune.py --epochs 100 --output_dir "RGZ/ViTB/100p" --dataset_train $data_path --ims_per_batch 16  --eid 0 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --lr 0.000005

#RN50
srun python  finetune.py --epochs 100 --output_dir "RGZ/rn50/10p" --dataset_train $data_path --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 469

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn50/30p" --dataset_train $data_path --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 1408

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn50/50p" --dataset_train $data_path --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" --seed 14 --nlabels 2346

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn50/100p" --dataset_train $data_path --ims_per_batch 16  --eid 3 --use_fp16 --model_name "microsoft/resnet-50" 

#RN18
srun python  finetune.py --epochs 100 --output_dir "RGZ/rn18/10p" --dataset_train $data_path --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 469

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn18/30p" --dataset_train $data_path --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 1408

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn18/50p" --dataset_train $data_path --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15 --nlabels 2346

srun python  finetune.py --epochs 100 --output_dir "RGZ/rn18/100p" --dataset_train $data_path --ims_per_batch 16  --eid 1 --use_fp16 --model_name "microsoft/resnet-18" --seed 15