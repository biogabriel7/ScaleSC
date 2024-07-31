#!/bin/bash 
#SBATCH --job-name=harm
#SBATCH -p gpu
#SBATCH -t 50-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=6 --mem=700G
#SBATCH --gres=gpu:A100:1
#SBATCH -o gpu.out
#SBATCH -e gpu.err
################# 
module load anaconda3 
conda activate /edgehpc/dept/compbio/users/whu1/envs/gpu_scanpy
python gpu_demo.py

#### python calculate_df.marker.CPM.csv_extra.py --celltype $SLURM_ARRAY_TASK_ID 
################# 
conda deactivate
