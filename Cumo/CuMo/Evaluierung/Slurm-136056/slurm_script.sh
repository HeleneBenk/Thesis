#!/bin/bash
#!/bin/bash
#SBATCH --job-name=cumoEval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benkehel@students.zhaw.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=earth-4
#SBATCH --qos=earth-4.2d
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=50G

export CUDA_HOME=/cfs/earth/scratch/benkehel/cuda12.1/toolkit
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cfs/earth/scratch/benkehel/.conda/envs/cumoTorch21/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

export TRANSFORMERS_CACHE=/cfs/earth/scratch/benkehel/hf_cache
export HF_HOME=/cfs/earth/scratch/benkehel/hf_cache


export PYTHONPATH=/cfs/earth/scratch/benkehel/CuMo:$PYTHONPATH
export CuMo_DIR=/cfs/earth/scratch/benkehel/CuMo
export TRITON_CACHE_DIR=/cfs/earth/scratch/benkehel/.triton_cache

python /cfs/earth/scratch/benkehel/CuMo/cumo/eval/model_vqa_loader.py \
  --model-path /cfs/earth/scratch/benkehel/CuMo/checkpoints/cumo-mistral-7b-sft \
  --model-base /cfs/earth/scratch/benkehel/CuMo/checkpoints/CuMo-mistral-7b \
  --image-folder /cfs/earth/scratch/benkehel/CuMo/data/Fertignummeriert \
  --question-file /cfs/earth/scratch/benkehel/CuMo/data/jsons/questionV2.jsonl \
  --answers-file /cfs/earth/scratch/benkehel/CuMo/data/jsons/outputV2_1200.jsonl \
  --conv-mode llava_v1 \
  --temperature 0.0 \
  --num_beams 1 \
  --max_new_tokens 128

