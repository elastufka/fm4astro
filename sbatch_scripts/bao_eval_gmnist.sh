#!/bin/sh
#SBATCH --job-name eval_GMNIST            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/eval_GMNIST-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/eval_GMNIST-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=20G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 02:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --exclude=gpu035

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92

srun python /home/users/l/lastufka/FM_compare/downstream.py --config /home/users/l/lastufka/FM_compare/eval_rgz.yaml --vary_inputs # --n_labels 628 2096 3144