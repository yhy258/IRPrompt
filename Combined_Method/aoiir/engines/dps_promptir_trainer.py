import os
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from diffusers import AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from aoiir.datasets.multi_degradation import MultiDegradationDataset
from aoiir.models import LatentPromptAdapter, PromptAwareEncoder
from aoiir.pipelines.dps_promptir import DPSPromptIRPipeline
from aoiir.metrics.metrics import psnr_ssim, lpips as lpips_metric, dists as dists_metric


def to01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class DPSPromptIRModule(pl.LightningModule):
    def __init__(
        self,
        sd_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        lr: float = 1e-4,
        lambda_latent: float = 1.0,
        lambda_pixel: float = 1.0,
        sample_mode: str = "posterior",  # "posterior" | "adapter_posterior"
        zeta_min: float = 0.05,
        zeta_max: float = 0.2,
        num_steps: int = 30,
        image_log_interval: int = 1,
        degrade_types: Optional[List[str]] = None,
        data_root: str = "/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset",
        data_file_dir: str = "/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/data_dir",
        batch_size: int = 2,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.pipeline = DPSPromptIRPipeline(sd_model_name=sd_model_name, adapter=None, lambda_latent=lambda_latent, lambda_pixel=lambda_pixel, device=self.device if self.device else "cuda")

        # Freeze SD modules already inside pipeline
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet

        self.lr = lr
        self.num_steps = num_steps
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.image_log_interval = image_log_interval
        self.lambda_latent = lambda_latent
        self.lambda_pixel = lambda_pixel
        assert sample_mode in ("posterior", "adapter_posterior")
        self.sample_mode = sample_mode

    def forward(self, x_D, x_G):
        z_D, z_G, total_loss,latent_loss, pixel_loss = self.pipeline(x_D, x_G)
        return z_D, z_G, total_loss,latent_loss, pixel_loss

    def training_step(self, batch, batch_idx):
        x_D, x_G = batch  # [-1,1]
        # Ensure pipeline modules are on the same device as LightningModule
        if str(self.pipeline.device) != str(self.device):
            self._set_pipeline_device(self.device)
        
        z_D, z_G, loss,latent_loss, pixel_loss = self(x_D, x_G)
        self.log("train/l2_latent", latent_loss, prog_bar=True)
        self.log("train/l2_pixel", pixel_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def _metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out.update(psnr_ssim(pred, gt))
        out["lpips"] = lpips_metric(pred, gt)
        out["dists"] = dists_metric(pred, gt)
        return out

    def validation_step(self, batch, batch_idx):
        x_D, x_G = batch
        if str(self.pipeline.device) != str(self.device):
            self._set_pipeline_device(self.device)
        with torch.no_grad():
            z_D, z_G, loss,latent_loss, pixel_loss = self(x_D, x_G)

        x_direct = self.pipeline.decode(z_G)
        m1 = self._metrics(x_direct, x_G)
        self.log("val/psnr_direct", m1.get("psnr"), prog_bar=True)
        self.log("val/ssim_direct", m1.get("ssim"), prog_bar=True)
        self.log("val/lpips_direct", m1.get("lpips"))
        self.log("val/dists_direct", m1.get("dists"))

        # DPS sampling
        z_rec = self.pipeline.dps_sample(z_D, num_steps=self.num_steps, zeta_min=self.zeta_min, zeta_max=self.zeta_max)
        x_rec = self.pipeline.decode(z_rec)
        m2 = self._metrics(x_rec, x_G)
        self.log("val/psnr_dps", m2.get("psnr"), prog_bar=True)
        self.log("val/ssim_dps", m2.get("ssim"), prog_bar=True)
        self.log("val/lpips_dps", m2.get("lpips"))
        self.log("val/dists_dps", m2.get("dists"))

        if (self.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):
            n = min(4, x_D.size(0))
            out_dir = f"validation_samples_dps/epoch_{self.current_epoch}-{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            grid = []
            for i in range(n):
                grid.extend([to01(x_D[i]), to01(x_G[i]), to01(x_direct[i]), to01(x_rec[i])])
            save_image(grid, f"{out_dir}/degraded_clean_direct_dps.png", nrow=4)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.pipeline.adapter.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return {"optimizer": opt, "lr_scheduler": sch}

    def _set_pipeline_device(self, device):
        self.pipeline.device = device
        self.pipeline.vae.to(device)
        self.pipeline.unet.to(device)
        self.pipeline.text_encoder.to(device)
        self.pipeline.adapter.to(device)

    def on_fit_start(self):
        self._set_pipeline_device(self.device)

    def on_validation_start(self):
        self._set_pipeline_device(self.device)


class MultiDegDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 2, num_workers: int = 4, patch_size: int = 256, val_ratio: float = 0.05):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.val_ratio = float(val_ratio)

    def setup(self, stage=None):
        full = MultiDegradationDataset(dataset_root=self.data_root, patch_size=self.patch_size)
        n_total = len(full)
        n_val = max(1, int(n_total * self.val_ratio))
        n_train = max(1, n_total - n_val)
        self.train_set, self.val_set = random_split(full, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


