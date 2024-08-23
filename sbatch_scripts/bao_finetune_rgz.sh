#!/bin/sh
#SBATCH --job-name finetune_RGZ            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/finetune_RGZ-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/finetune_RGZ-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu009 #5,gpu012,gpu013 #,gpu006 #,gpu014,gpu013,gpu015,gpu016 #,gpu006 #,gpu007,gpu012,gpu013,gpu014,gpu015 #,gpu015,gpu016,gpu036 #9,gpu014,gpu020,gpu025,gpu026,gpu035 #,gpu036 #,gpu011,gpu012,gpu014

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="supervised_RGZ"
#srun python /home/users/l/lastufka/FM_compare/detection.py --backbone

# supervised ViTB/16
#RN50
srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/RGZ/supervised_single100p" --ims_per_batch 16  --eid 2 --use_fp16 --model_name "google/vit-base-patch16-224"  --seed 14 --lr 0.000001 --dataset_train /home/users/l/lastufka/scratch/rgz #--nlabels 469

#DINOv2
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/RGZ/dinov2" --ims_per_batch 16  --eid 4 --freeze_backbone --use_fp16 --dataset_train /home/users/l/lastufka/scratch/rgz

#MSN
# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/RGZ/msn" --ims_per_batch 16  --eid 5 --freeze_backbone --use_fp16 --model_name "facebook/vit-msn-base" --dataset_train /home/users/l/lastufka/scratch/rgz --lr 0.00002

#MAE
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/RGZ/mae" --ims_per_batch 16  --eid 6 --freeze_backbone --use_fp16 --model_name "facebook/vit-mae-base" --dataset_train /home/users/l/lastufka/scratch/rgz #-resume

#RN50
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50" --ims_per_batch 16  --eid 7 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --dataset_train /home/users/l/lastufka/scratch/rgz

#MGCLS ViTB
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/vit_det_mgcls_transfer_30p" --ims_per_batch 16  --eid 0 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224" --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth  --seed 15 --nlabels 2096

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/vit_det_mgcls_transfer_50p" --ims_per_batch 16  --eid 1 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224" --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth  --seed 15 --nlabels 3144

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/vit_det_mgcls_transfer_100p" --ims_per_batch 16  --eid 9 --freeze_backbone --use_fp16 --model_name "google/vit-base-patch16-224" --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth  --seed 15
#--nlabels 628 
#GZ2 RN50
# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_gz2_transfer_10p" --ims_per_batch 16  --eid 24 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 628 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth  --seed 15

# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_gz2_transfer_30p" --ims_per_batch 16  --eid 25 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 2096 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth  --seed 15

# srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_gz2_transfer_50p" --ims_per_batch 16  --eid 26 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --nlabels 3144 --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth  --seed 15

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_gz2_transfer_100p" --ims_per_batch 16  --eid 27 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth  --seed 15 --resume

#RGZ RN18
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_rgz_transfer_100p" --ims_per_batch 16  --eid 10 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18" --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth  --seed 15 #--nlabels 628 

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_rgz_transfer_30p" --ims_per_batch 16  --eid 11 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18" --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth  --seed 15 --nlabels 2096 

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_rgz_transfer_50p" --ims_per_batch 16  --eid 12 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18" --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth  --seed 15 --nlabels 3144 

#srun python /home/users/l/lastufka/FM_compare/finetune.py --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir /home/users/l/lastufka/FM_compare/RGZ/supervised_30p_nonorm --lr 0.00005 --nlabels 2096 --seed 15 --eid 1

#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/mae_transfer_10p" --ims_per_batch 16  --eid 31 --freeze_backbone --use_fp16 --model_name "facebook/vit-mae-base" --nlabels 628 --seed 15  --lr 0.0005

#srun python /home/users/l/lastufka/FM_compare/finetune.py --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir /home/users/l/lastufka/FM_compare/RGZ/supervised_30p --lr 0.00005 --nlabels 2096 --seed 15

#srun python /home/users/l/lastufka/FM_compare/finetune.py --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir /home/users/l/lastufka/FM_compare/RGZ/supervised_50p --lr 0.00005 --nlabels 3144 --seed 15 --eid 2

#srun python /home/users/l/lastufka/FM_compare/finetune.py --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir /home/users/l/lastufka/FM_compare/RGZ/supervised_100p --lr 0.00001 --eid 3-

## MGCLS RN50
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn50_mgcls_transfer_100p_lr0.000001" --ims_per_batch 16  --eid 4 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-50" --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_hf.pth  --seed 15 --lr 0.000001

##RN18
#srun python /home/users/l/lastufka/FM_compare/finetune.py --epochs 100 --dataset_train /home/users/l/lastufka/scratch/rgz --output_dir "/home/users/l/lastufka/FM_compare/RGZ/rn18" --ims_per_batch 16  --eid 0 --freeze_backbone --use_fp16 --model_name "microsoft/resnet-18" --seed 15
