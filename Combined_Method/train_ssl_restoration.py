import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from aoiir.engines.ssl_unet_train import SSLUNetRestoration, SSLDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train SSL-conditioned UNet restoration (SD backbone)")
    # data
    parser.add_argument("--data_root", type=str, required=True, help="Root dir containing datasets (e.g., Combined_Method/datasets)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--use_multi_degradation", action="store_true", 
                       help="Use multi-degradation dataset (denoise, dehaze, deblur, etc.)")
    parser.add_argument("--degradation_types", type=str, nargs="+", 
                       default=["denoise_15", "dehaze", "deblur", "lowlight"],
                       help="Types of degradation to include")
    # model
    parser.add_argument("--sd_model_name", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--ssl_model_name", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--num_ssl_tokens", type=int, default=8)
    parser.add_argument("--clip_hidden", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--init_strength", type=float, default=0.8)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--train_cross_attn_subset", action="store_true")
    # trainer
    parser.add_argument("--output_dir", type=str, default="ssl_runs")
    parser.add_argument("--max_steps", type=int, default=0, help="Use >0 to limit by steps")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit fraction/num of train batches (e.g., 0.1 or 10)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.limit_train_batches > 1.0:
        args.limit_train_batches = int(args.limit_train_batches)

    # Data
    data = SSLDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        use_sd_scaling=True,
        use_multi_degradation=args.use_multi_degradation,
        degradation_types=args.degradation_types,
    )

    # Model
    model = SSLUNetRestoration(
        sd_model_name=args.sd_model_name,
        ssl_model_name=args.ssl_model_name,
        num_ssl_tokens=args.num_ssl_tokens,
        clip_hidden=args.clip_hidden,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        init_strength=args.init_strength,
        lambda_align=args.lambda_align,
        train_cross_attn_subset=args.train_cross_attn_subset,
    )

    # Trainer
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="ssl-unet-{epoch:03d}-{step:06d}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=None if args.max_steps > 0 else args.max_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        log_every_n_steps=10,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()


