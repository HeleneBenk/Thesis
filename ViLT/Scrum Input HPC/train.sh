#!/bin/bash

#SBATCH --job-name=ViLTV1
#SBATCH --mail-type=all              
#SBATCH --mail-user=benkehel@students.zhaw.ch
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=32            
#SBATCH --time=4-00:00:00
#SBATCH --mem=64GB                   
#SBATCH --partition=earth-3


module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load lsfm-init-miniconda/1.0.0
module load cuda/11.6.2
conda activate myenv

python /cfs/earth/scratch/benkehel/ViLT/V1/ViLT_V1_Train.py
