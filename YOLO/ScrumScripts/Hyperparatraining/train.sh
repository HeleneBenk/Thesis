#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --mail-type=all              
#SBATCH --mail-user=benkehel@students.zhaw.ch
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=32            
#SBATCH --time=2-00:00:00
#SBATCH --mem=100GB                   
#SBATCH --partition=earth-4
#SBATCH --qos=earth-4.2d
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:l40s:1

conda activate yolo11

export WANDB_API_KEY=NO_WANDB
export WANDB_DISABLED=true


export TMPDIR=/cfs/earth/scratch/benkehel/tmp

python yolo_ray_tune.py
