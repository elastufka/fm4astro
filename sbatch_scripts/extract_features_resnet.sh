#!/bin/sh 
#SBATCH --job-name extract_feats            # this is a parameter to help you sort your job when listing it
#SBATCH --error sbatch_logs/extract_feats-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output sbatch_logs/extract_feats-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --mem=50G
#SBATCH --gpus 1
#SBATCH --time 01:00:00                  # maximum run time.

data_path=/path/to/rgz
dump_path=/path/to/output/RGZ

#RN50
srun python eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1  --in_chans 3 --resize 256 --center_crop 256

mv $dump_path/trainfeat.pth $dump_path/RGZ_trainfeat_RN50.pth

srun python eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1 --train false  --in_chans 3 --center_crop 256 --resize 256

mv $dump_path/trainfeat.pth $dump_path/RGZ_testfeat_RN50.pth

#RN18
srun python eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1  --in_chans 3 --resize 256 --center_crop 256

mv $dump_path/trainfeat.pth $dump_path/RGZ_trainfeat_RN18.pth

srun python eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1 --train false  --in_chans 3 --center_crop 256 --resize 256

mv $dump_path/trainfeat.pth $dump_path/RGZ_testfeat_RN18.pth
