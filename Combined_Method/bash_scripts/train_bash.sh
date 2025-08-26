# Set GPU devices
# export CUDA_VISIBLE_DEVICES=5

# Train E_D (degraded image encoder alignment)
# cd /home/joon/ImageRestoration-AllInOne/Combined_Method

# python train_ssl_restoration.py \
#   --data_root /home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset \
#   --batch_size 64 --num_workers 4 --crop_size 256 \
#   --sd_model_name stabilityai/stable-diffusion-2-1-base \
#   --ssl_model_name vit_base_patch14_reg4_dinov2.lvd142m \
#   --num_ssl_tokens 16 --clip_hidden 1024 \
#   --learning_rate 1e-4 --diffusion_steps 250 \
#   --guidance_scale 1.0 --init_strength 0.8 --lambda_align 1.0 \
#   --train_cross_attn_subset \
#   --use_multi_degradation \
#   --degradation_types denoise_15 denoise_25 denoise_50 derain dehaze deblur lowlight \
#   --output_dir ssl_runs2 --max_epochs 100 \
#   --accelerator gpu

export CUDA_VISIBLE_DEVICES=4

python dps_promptir_train.py \
  --data_root /home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset \
  --sd_model stabilityai/stable-diffusion-2-1-base \
  --batch_size 16 --epochs 50 --steps 50 --lr 0.0002 \
  --val_ratio 0.005 --val_interval_steps 1000 --gpus 1 \
  --lambda_latent 1.0 --lambda_pixel 1.0 \
