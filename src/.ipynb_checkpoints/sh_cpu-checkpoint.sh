#!/bin/bash 
#SBATCH --job-name=harm
#SBATCH -p cpu
#SBATCH -t 50-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=6 --mem=256G
####SBATCH --gres=gpu:1
####SBATCH --nodelist=edge-hpc-gpu-116
#SBATCH -a 0
#SBATCH -o cpu.out
#SBATCH -e cpu.err
################# 
module load anaconda3 
conda activate /edgehpc/dept/compbio/users/whu1/envs/scanpy_1.9
python cpu_demo.py  --celltype $SLURM_ARRAY_TASK_ID 
#### --celltype $SLURM_ARRAY_TASK_ID 
#### python calculate_df.marker.CPM.csv_extra.py --celltype $SLURM_ARRAY_TASK_ID 
################# 
conda deactivate
