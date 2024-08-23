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
SBATCH --exclude=gpu002
#conda activate torch            # load a specific software using module, for example Python
#ml GCC/11.3.0
#ml OpenMPI/4.1.4
#ml PyTorch/1.12.1-CUDA-11.7.0 

data_path=/home/users/l/lastufka/scratch/decals
dump_path=/home/users/l/lastufka/FM_compare/GZ10

#data_path=/home/users/l/lastufka/scratch/rgz/od
#dump_path=/home/users/l/lastufka/FM_compare/RGZ
#srun python ~/mightee_dino/main_stix.py --data_path ~/scratch/MGCLS_data/enhanced/small_crops --output_dir ~/scratch/MGCLS_data/small_crops/ViT-S16/ --arch vit_small --epochs 500  --saveckp_freq 100 --num_workers 0 --batch_size_per_gpu 16 --inchans 3          
# ls /home/users/l/lastufka/scratch/GalaxyMNIST
# #extract features to trainfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_3chan.pth --world_size 1 --in_chans 3 --resize 256 --center_crop 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_mgRN50_425b_trainfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/DINO/mgcls_resnet50_pretrain_425b_3chan.pth --world_size 1  --train false --in_chans 3 # --resize 256 --center_crop 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_mgRN50_425b_testfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_3chan_teacher.pth --world_size 1  --in_chans 3 --center_crop 256 --resize 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_gzRN50_trainfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet50 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/hayat_gz/mocov2_encoder_k_3chan_teacher.pth --world_size 1 --train false  --in_chans 3 --center_crop 256 --resize 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_gzRN50_testfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/byol/byol_3chan_teacher.pth --world_size 1  --in_chans 3 --resize 256 

# mv $dump_path/trainfeat.pth $dump_path/GZ10_rgzRN18_trainfeat.pth

srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --pretrained_weights /home/users/l/lastufka/byol/byol_3chan_teacher.pth --world_size 1 --train false  --in_chans 3 #--resize 256

mv $dump_path/trainfeat.pth $dump_path/GZ10_rgzRN18_testfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1  --in_chans 3 --resize 256 --center_crop 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_tvRN18_trainfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch resnet18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1 --train false  --in_chans 3 --center_crop 256 --resize 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_tvRN18_testfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch customRN18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1  --in_chans 3 --pretrained_weights /home/users/l/lastufka/riggi-SimCLR/encoder_weights-resnet18_simclr_smarthulk256-smgps_ch3_500epochs.pth --resize 256 

# mv $dump_path/trainfeat.pth $dump_path/GZ10_shRN18_trainfeat.pth

# srun python /home/users/l/lastufka/mightee_dino/eval_knn_train.py --arch customRN18 --patch_size 8 --dump_features $dump_path --data_path $data_path --batch_size_per_gpu 16 --num_workers 0 --world_size 1 --train false  --in_chans 3 --pretrained_weights /home/users/l/lastufka/riggi-SimCLR/encoder_weights-resnet18_simclr_smarthulk256-smgps_ch3_500epochs.pth --resize 256

# mv $dump_path/trainfeat.pth $dump_path/GZ10_shRN18_testfeat.pth


#srun python ~/mightee_dino/main_meerkat.py --data_path ~/scratch/MIGHTEE/early_science/arrays/field_normalized --output_dir ~/scratch/MIGHTEE/early_science/ViT-S16/ --arch vit_small --epochs 100 --saveckp_freq 50 --num_workers 0 --batch_size_per_gpu 16 --inchans 3 --lr 0.0002         --pretrained_weights ~/scratch/MIGHTEE/early_science/arrays/adapteq/ViT-S16/checkpoint.pth

