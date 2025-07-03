#!/bin/bash
#SBATCH --job-name=yoloEval
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


yolo detect predict \
  model=/cfs/earth/scratch/benkehel/YOLO/ScrumScripts/runs/detect/abfall_cpu/weights/best.pt \
  source=/cfs/earth/scratch/benkehel/YOLO/test \
  save=True \
  save_txt=True \
  save_conf=True \
  project=/cfs/earth/scratch/benkehel/YOLO/runs \
  name=test_inference

yolo detect val \
  model=/cfs/earth/scratch/benkehel/YOLO/ScrumScripts/runs/detect/abfall_cpu/weights/best.pt \
  data=/cfs/earth/scratch/benkehel/YOLO/dataset.yaml \
  split=test
