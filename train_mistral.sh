# Run this script: nohup bash weighted.sh > train_3.0.log &
RUN_NAME=gumbel_scale_100_delta_4.0

echo "Running training: $RUN_NAME"

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 train_mistral.py \
CUDA_VISIBLE_DEVICES=1 python train_mistral.py \
    --model_name_or_path=/data2/mingjia/Mistral-7B-Instruct-v0.2 \
    --model_STS_name_or_path=/data2/mingjia/SFR-Embedding-2_R \
    --dtype=fp16 \
    --fix_gamma=False \
    --scale=100 \
    --max_new_tokens=100 \
    --prompt_tokens=100 \
    --run_name=$RUN_NAME \
    --wandb=False \
    --verbose=False \
    --batch_size=8 \
    --ckpt_dir=ckpt \
    --lr=3e-5 \
    --adaptor=LSTM \
    --layer_gamma=2 \
    --layer_delta=2 \
    --init_val_gamma=0.1 \
    --init_val_delta=4.0 \
    --log_z_score=False \
    --z_score_factor=2e-4 \
    --optimizer=Adam \
    --epochs=2 \
    --log_ckpt=True \
    --log_freq=200 \
# Notes:
# layer_gamma and layer_delta are only used when adaptor=MLP 