#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --mail-type=all              
#SBATCH --mail-user=benkehel@students.zhaw.ch
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=32            
#SBATCH --time=4-00:00:00
#SBATCH --mem=100GB                   
#SBATCH --partition=earth-3

conda activate yolo11

export TMPDIR=/cfs/earth/scratch/benkehel/tmp
export YOLO_CONFIG_DIR=/cfs/earth/scratch/benkehel/ultralytics_config

yolo detect train \
  model=yolo11m.pt \
  data=/cfs/earth/scratch/benkehel/YOLO/dataset.yaml \
  epochs=200 \
  imgsz=640 \
  batch=8 \
  name=abfall_cpu
