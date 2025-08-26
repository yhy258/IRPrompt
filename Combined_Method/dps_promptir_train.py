import argparse
import pytorch_lightning as pl
import torch

from aoiir.engines.dps_promptir_trainer import DPSPromptIRModule, MultiDegDataModule
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset')
    parser.add_argument('--sd_model', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_latent', type=float, default=1.0)
    parser.add_argument('--lambda_pixel', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--zeta_min', type=float, default=0.05)
    parser.add_argument('--zeta_max', type=float, default=0.2)
    parser.add_argument('--val_interval_steps', type=int, default=None, help='Run validation every N training steps (int).')
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Fraction of data used for validation split.')
    parser.add_argument('--sample_mode', type=str, default='posterior', choices=['posterior','adapter_posterior'], help='How to sample latents during training.') # 그냥 무조건 posterior로 해야 하는게 신상에 좋다.
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    dm = MultiDegDataModule(data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, val_ratio=args.val_ratio)
    model = DPSPromptIRModule(sd_model_name=args.sd_model, lr=args.lr, num_steps=args.steps, zeta_min=args.zeta_min, zeta_max=args.zeta_max, lambda_latent=args.lambda_latent, lambda_pixel=args.lambda_pixel, sample_mode=args.sample_mode)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        precision=32,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir='runs/dps_promptir',
        check_val_every_n_epoch=1,
        val_check_interval=args.val_interval_steps if args.val_interval_steps is not None else 1.0,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()


