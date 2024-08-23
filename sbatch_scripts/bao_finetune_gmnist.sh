#!/bin/sh
#SBATCH --job-name finetune_GMNIST            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/finetune_GMNIST-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/finetune_GMNIST-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu008,gpu006,gpu007 #4,gpu005,gpu013 #,gpu011,gpu012,gpu013,gpu014,gpu043

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="supervised_GMNIST"
#srun python /home/users/l/lastufka/FM_compare/detection.py --backbone
#pip install transformers
# supervised ViTB/16
#RN50
srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/RN50supervised10p" --ims_per_batch 16  --eid 0 --use_fp16 --model_name "microsoft/resnet-50"  --seed 14 --nlabels 800 --lr 0.000005

#DINOv2
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/dinov2" --ims_per_batch 16  --eid 0 --freeze_backbone --use_fp16 

#MSN
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/msn" --ims_per_batch 16  --eid 1 --freeze_backbone --use_fp16 --model_name "facebook/vit-msn-base"

#MAE
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/mae" --ims_per_batch 16  --eid 2 --freeze_backbone --use_fp16 --model_name "facebook/vit-mae-base"

#RN50
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50" --ims_per_batch 16  --eid 3 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50"

#MGCLS ViTB
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/vit_det_mgcls_transfer_100p" --ims_per_batch 16  --eid 6 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224"  --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth #--nlabels 800

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/vit_det_mgcls_transfer_30p" --ims_per_batch 16  --eid 13 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224"  --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth --nlabels 2400

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/vit_det_mgcls_transfer_50p" --ims_per_batch 16  --eid 14 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224"  --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth --nlabels 4000


#GZ2 RN50
# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_gz2_transfer_10p" --ims_per_batch 16  --eid 20 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 800 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth 

# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_gz2_transfer_30p" --ims_per_batch 16  --eid 21 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 2400 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth 

# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_gz2_transfer_50p" --ims_per_batch 16  --eid 22 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 4000 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth 

# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_gz2_transfer_100p" --ims_per_batch 16  --eid 23 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth 

#RGZ RN18
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_rgz_transfer_10p" --ims_per_batch 16  --eid 7 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18"  --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth --nlabels 800

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_rgz_transfer_30p" --ims_per_batch 16  --eid 15 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18"  --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth --nlabels 2400

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_rgz_transfer_50p" --ims_per_batch 16  --eid 16 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18"  --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth --nlabels 4000

## MGCLS RN50
# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn50_mgcls_transfer_100p_lr0.0001" --ims_per_batch 16  --eid 2 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_hf.pth  --seed 15 --lr 0.0001

##RN18
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/GMNIST/rn18" --ims_per_batch 16  --eid 1 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18" --seed 15
