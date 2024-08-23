#!/bin/sh
#SBATCH --job-name eval_frcnn            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/eval_frcnn-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/eval_frcnn-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=50G
##SBATCH --gres=gpu:2,VramPerGpu:20G
#SBATCH --time 06:00:00                  # maximum run time.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu009,gpu008

#export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
#export WANDB_PROJECT="Faster-RCNN"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29600'

folders=$(find "/home/users/l/lastufka/outputs/training/" -type d -name 'resnet18*') #/*/)
#echo $folders
for f in $folders; do #[@]
    f=${f%/}
    opt_file="$f/opt.yaml"

    # Check if the opt.yaml file exists
    #if [[ -f "$opt_file" ]]; then
    # Get the line that starts with 'data:'
    data_line=$(grep '^data:' "$opt_file")
    data=${data_line#*: }

    model_line=$(grep '^model:' "$opt_file")
    model=${model_line#*: }
    
    if [[ $data == *"MGCLS"* ]]; then
      imgsz=512
    else
      imgsz=256  # Set a default value if needed
    fi
    
    echo $f
    echo $data
    
    #srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/train_frcnn.py --data $data --weights $f/last_model.pth --model $model --imgsz $imgsz --eval_only --batch 16  #--thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
    srun python /home/users/l/lastufka/fastercnn-pytorch-training-pipeline/eval.py --data $data --weights $f/best_model.pth --model $model --imgsz $imgsz --thresholds 0.2 0.5 --verbose

done
