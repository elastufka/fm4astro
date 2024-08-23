#!/bin/sh
#SBATCH --job-name extract_feats_RGZ            # this is a parameter to help you sort your job when listing it
#SBATCH --error  sbatch_logs/extract_feats_RGZ-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output  sbatch_logs/extract_feats_RGZ-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=100G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL

#pip install transformers

data_path=/path/to/rgz
dname=RGZ

if [ ! -f "$dname"_testfeat_dinov2B.pth ]; then
    srun python extract_features.py --dataset_train $data_path --test --img_fmt PIL
    mv features.pth "$dname"_testfeat_dinov2B.pth
fi

if [ ! -f "$dname"_trainfeat_dinov2B.pth ]; then
    srun python extract_features.py --dataset_train $data_path --img_fmt PIL
    mv features.pth "$dname"_trainfeat_dinov2B.pth
fi

if [ ! -f "$dname"_trainfeat_maeB.pth ]; then
    srun python extract_features.py --dataset_train $data_path --model facebook/vit-mae-base --img_fmt PIL
    mv features.pth "$dname"_trainfeat_maeB.pth
fi

if [ ! -f "$dname"_testfeat_maeB.pth ]; then 
    srun python extract_features.py --dataset_train  $data_path --model facebook/vit-mae-base --test --img_fmt PIL
    mv features.pth "$dname"_testfeat_maeB.pth
fi

if [ ! -f "$dname"_trainfeat_msnB.pth ]; then
    srun python extract_features.py --dataset_train  $data_path --model facebook/vit-msn-base --img_fmt PIL
    mv features.pth "$dname"_trainfeat_msnB.pth
fi

if [ ! -f "$dname"_testfeat_msnB.pth ]; then
    srun python extract_features.py --dataset_train $data_path --model facebook/vit-msn-base --test --img_fmt PIL
    mv features.pth "$dname"_testfeat_msnB.pth
fi
