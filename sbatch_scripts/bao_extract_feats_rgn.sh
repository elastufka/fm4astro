#!/bin/sh
#SBATCH --job-name extract_feats_HF            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/extract_feats_HF-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/extract_feats_HF-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 05:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu008

pip install transformers
export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92

data_path=/home/users/l/lastufka/scratch/rgz/od #/home/users/l/lastufka/scratch/RadioGalaxyNET
dname=RGZ #RGN

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_dinov2B.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --test --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_dinov2B.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_dinov2B.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_dinov2B.pth
# fi

if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_sam.pth ]; then
    srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --test --img_fmt PIL --model google/vit-base-patch16-224-in21k --weights /home/users/l/lastufka/segment-anything/SAM_pretrained_state_dict.pth
    mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_sam.pth
fi

if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_sam.pth ]; then
    srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --img_fmt PIL --model google/vit-base-patch16-224-in21k --weights /home/users/l/lastufka/segment-anything/SAM_pretrained_state_dict.pth
    mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_sam.pth
fi

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_dinov1B.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --test --img_fmt PIL --model facebook/dino-vitb16 
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_dinov1B.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_dinov1B.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --img_fmt PIL --model facebook/dino-vitb16
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_dinov1B.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_trainfeat_maeB.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model facebook/vit-mae-base --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_maeB.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_testfeat_maeB.pth ]; then 
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train  $data_path --model facebook/vit-mae-base --test --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_maeB.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_trainfeat_msnB.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train  $data_path --model facebook/vit-msn-base --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_msnB.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_testfeat_msnB.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model facebook/vit-msn-base --test --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_msnB.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_trainfeat_RN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_RN50.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_testfeat_RN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --test --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_RN50.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_trainfeat_RN152.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-152 --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_RN152.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_testfeat_RN152.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-152 --test --img_fmt PIL --resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_RN152.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_trainfeat_msnB4.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model facebook/vit-msn-base-4 --img_fmt PIL #--resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_msnB4.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/RGZ/"$dname"_testfeat_msnB4.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model facebook/vit-msn-base-4 --test --img_fmt PIL #--resize 256
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_msnB4.pth
# fi

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_mgclsRN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --img_fmt PIL  --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_teacher.pth
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_mgclsRN50.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_mgclsRN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --img_fmt PIL  --weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_teacher.pth --test
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_mgclsRN50.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_gzRN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --img_fmt PIL  --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_gzRN50.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_gzRN50.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-50 --img_fmt PIL  --weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_hf.pth --test
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_gzRN50.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_mgclsViTB16.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model google/vit-base-patch16-224 --img_fmt PIL  --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_mgclsViTB16.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_mgclsViTB16.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model google/vit-base-patch16-224 --img_fmt PIL  --weights /home/users/l/lastufka/DINO/mgcls_vitbase16_finetune_teacher_025_hf.pth --test
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_mgclsViTB16.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_rgzRN18.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-18 --img_fmt PIL  --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_trainfeat_rgzRN18.pth
# fi 

# if [ ! -f /home/users/l/lastufka/FM_compare/"$dname"_testfeat_rgzRN18.pth ]; then
#     srun python /home/users/l/lastufka/FM_compare/extract_features.py --dataset_train $data_path --model microsoft/resnet-18 --img_fmt PIL  --weights /home/users/l/lastufka/byol/byol_resnet18_hf.pth --test
#     mv /home/users/l/lastufka/FM_compare/features.pth /home/users/l/lastufka/FM_compare/"$dname"_testfeat_rgzRN18.pth
# fi 
