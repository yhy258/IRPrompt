import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


# Ensure local package is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aoiir.engines.encoder_alignment import EncoderAlignmentModel, AlignmentDataModule


def main():
    # Speed-oriented defaults
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_autoencoder_path", type=str, default='none',
                        help="Path to the pretrained autoencoder checkpoint")
    parser.add_argument("--data_root", type=str, default="./dataset",
                        help="Path to the root of the datasets")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--similarity_weight", type=float, default=1.0)
    parser.add_argument("--l2_weight", type=float, default=1.0)
    parser.add_argument("--img_loss_weight", type=float, default=0.5)
    parser.add_argument("--init_strength", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--supir_sampling_enabled", action="store_true",
                        help="Use SUPIR sampling")
    parser.add_argument("--supir_noise_level", type=float, default=0.1)
    parser.add_argument("--supir_restoration_guidance", type=float, default=1.5)
    parser.add_argument("--color_correction_on", action="store_true",
                        help="Use color correction")
    parser.add_argument("--use_stable_diffusion", action="store_true",
                        help="Use Stable Diffusion VQVAE instead of custom autoencoder")
    parser.add_argument("--use_compvis_vqvae", action="store_true",
                        help="Use CompVis VQ-f4 model (31.40+ dB PSNR)")
    parser.add_argument("--compvis_model_path", type=str, default="model.ckpt",
                        help="Path to CompVis VQ-f4 model checkpoint")
    parser.add_argument("--sd_model_name", type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="Stable Diffusion model name")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_promptir", action="store_true")
    parser.add_argument("--image_log_interval", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=512)
    # No pair-list arguments (reverted)
    parser.add_argument("--precision", type=str, default="32",
                        help="Trainer precision")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--channels_last", action="store_true",
                        help="Use channels_last memory format for speed on Ampere+")
    # Iteration limiting for faster debug
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit fraction/num of train batches (e.g., 0.1 or 10)")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="Run validation every N train batches (int) or fraction of epoch (float)")
    opt = parser.parse_args()

    if opt.limit_train_batches > 1.0:
        opt.limit_train_batches = int(opt.limit_train_batches)


    data = AlignmentDataModule(
        data_root=opt.data_root,
        batch_size=opt.batch_size,
        use_compvis_vqvae=opt.use_compvis_vqvae,
        use_stable_diffusion=opt.use_stable_diffusion,
        num_workers=opt.num_workers,
        crop_size=opt.crop_size,
        
    )

    model = EncoderAlignmentModel(
        pretrained_autoencoder_path=opt.pretrained_autoencoder_path,
        learning_rate=opt.learning_rate,
        similarity_weight=opt.similarity_weight,
        l2_weight=opt.l2_weight,
        img_loss_weight=opt.img_loss_weight,
        use_stable_diffusion=opt.use_stable_diffusion,
        diffusion_steps=opt.diffusion_steps,
        use_compvis_vqvae=opt.use_compvis_vqvae,
        compvis_model_path=opt.compvis_model_path,
        sd_model_name=opt.sd_model_name,
        enable_promptir=(not opt.disable_promptir),
        image_log_interval=opt.image_log_interval,
        init_strength=opt.init_strength,
        guidance_scale=opt.guidance_scale,
        supir_sampling_enabled=opt.supir_sampling_enabled,
        supir_noise_level=opt.supir_noise_level,
        supir_restoration_guidance=opt.supir_restoration_guidance,
        color_correction_on=opt.color_correction_on,
        
    )

    if opt.channels_last:
        model = model.to(memory_format=torch.channels_last)

    os.makedirs("checkpoints/encoder_alignment", exist_ok=True)

    class EncoderCheckpointCallback(pl.Callback):
        def __init__(self, dirpath, filename_template, monitor="val/similarity", mode="max", save_top_k=3):
            self.dirpath = dirpath
            self.filename_template = filename_template
            self.monitor = monitor
            self.mode = mode
            self.save_top_k = save_top_k
            self.best_scores = []

        def on_validation_end(self, trainer, pl_module):
            current_score = trainer.callback_metrics.get(self.monitor)
            if current_score is not None:
                filename = self.filename_template.format(epoch=trainer.current_epoch, val_similarity=current_score.item())
                filepath = os.path.join(self.dirpath, f"{filename}.pth")
                torch.save({
                    'encoder_state_dict': pl_module.encoder_degraded.state_dict(),
                    'epoch': trainer.current_epoch,
                    'similarity': current_score.item(),
                    'use_stable_diffusion': pl_module.use_stable_diffusion,
                    'sd_model_name': getattr(pl_module, 'sd_model_name', 'stabilityai/stable-diffusion-2-1-base')
                }, filepath)
                self.best_scores.append((current_score.item(), filepath))
                if self.mode == "max":
                    self.best_scores.sort(key=lambda x: x[0], reverse=True)
                else:
                    self.best_scores.sort(key=lambda x: x[0])
                if len(self.best_scores) > self.save_top_k:
                    _, old_path = self.best_scores.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)
                print(f"Saved encoder checkpoint: {filepath}")

    callbacks = [
        EncoderCheckpointCallback(
            dirpath="checkpoints/encoder_alignment",
            filename_template="{epoch:02d}-{val_similarity:.4f}",
            monitor="val/similarity",
            mode="max",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=opt.max_epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        log_every_n_steps=50,
        accumulate_grad_batches=1,
        benchmark=True,
        precision=opt.precision,
        check_val_every_n_epoch=opt.val_every,
        limit_train_batches=opt.limit_train_batches,
        val_check_interval=opt.val_check_interval,
        num_sanity_val_steps=0,
    )

    print("Starting Encoder Alignment Training...")
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()




