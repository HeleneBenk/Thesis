#!/bin/bash
#!/bin/bash
#SBATCH --job-name=cumo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benkehel@students.zhaw.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --partition=earth-5
#SBATCH --qos=earth-5.1d
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=350G

export CUDA_HOME=/cfs/earth/scratch/benkehel/cuda12.1/toolkit
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cfs/earth/scratch/benkehel/.conda/envs/cumoTorch21/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

export HF_HOME=/cfs/earth/scratch/benkehel/huggingface
export TRANSFORMERS_CACHE=/cfs/earth/scratch/benkehel/huggingface
export HF_DATASETS_CACHE=/cfs/earth/scratch/benkehel/huggingface
export HF_METRICS_CACHE=/cfs/earth/scratch/benkehel/huggingface

# Torch C++ Extensions
export TORCH_EXTENSIONS_DIR=/cfs/earth/scratch/benkehel/torch_extensions

# Triton (FlashAttention Kernel)
export TRITON_CACHE_DIR=/cfs/earth/scratch/benkehel/.triton_cache



export PYTHONPATH=/cfs/earth/scratch/benkehel/CuMo:$PYTHONPATH
export CuMo_DIR=/cfs/earth/scratch/benkehel/CuMo

deepspeed $CuMo_DIR/cumo/train/train_mem.py \
    --deepspeed $CuMo_DIR/scripts/zero3_offload.json \
    --model_name_or_path $CuMo_DIR/checkpoints/CuMo-mistral-7b \
    --version mistral_instruct_system \
    --data_path $CuMo_DIR/data/jsons/train.json \
    --image_folder $CuMo_DIR/data/Fertignummeriert \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_dir $CuMo_DIR/checkpoints/CuMo-mistral-7b/clip.bin \
    --scales 1,3 \
    --pretrain_mm_mlp_adapter $CuMo_DIR/checkpoints/CuMo-mistral-7b/mm_projector.bin \
    --mm_projector_type smoe_mlp \
    --mlp_smoe True \
    --clip_smoe True \
    --num_experts 4 \
    --num_selected 2 \
    --balance_loss_coef 0.1 \
    --router_z_loss_coef 0.01 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --lr_scheduler_type cosine \
    --bf16 True \
    --fp16 False \
    --output_dir $CuMo_DIR/checkpoints/cumo-mistral-7b-sft \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 4e-6 \
    --weight_decay 0. \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
