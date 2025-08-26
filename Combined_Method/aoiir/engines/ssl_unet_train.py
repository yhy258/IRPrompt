import os
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from aoiir.datasets.paired import (
    DegradedCleanPairDataset,
    PairedRandomCropFlip,
    PairedCenterCrop,
)
from aoiir.datasets.multi_degradation import create_multi_degradation_dataset
from aoiir.models.ssl_backbone import SSLBackbone
from aoiir.models.ssl_adapter import SSLAdapter

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class SSLUNetRestoration(pl.LightningModule):
    def __init__(
        self,
        sd_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        ssl_model_name: str = "vit_base_patch14_dinov2",
        num_ssl_tokens: int = 8,
        clip_hidden: int = 1024,
        learning_rate: float = 1e-4,
        diffusion_steps: int = 1000,
        guidance_scale: float = 1.0,
        init_strength: float = 0.8,
        lambda_align: float = 0.1,
        train_cross_attn_subset: bool = False,
        # validation controls
        image_log_interval: int = 1,
        val_num_steps: int = 50,
        val_guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # SD components (frozen, except optional cross-attn subset)
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae", torch_dtype=torch.float32)
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet", torch_dtype=torch.float32)
        self.scheduler = DDIMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder", torch_dtype=torch.float32)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")

        freeze_module(self.vae)
        freeze_module(self.text_encoder)

        # Freeze UNet then unfreeze selective cross-attn
        freeze_module(self.unet)
        if train_cross_attn_subset:
            train_keys = [
                "mid_block.attentions",
                "up_blocks.0.attentions",
                "up_blocks.1.attentions",
            ]
            for n, p in self.unet.named_parameters():
                if any(k in n for k in train_keys):
                    p.requires_grad = True

        # SSL backbone (frozen) and adapter (trainable)
        self.ssl_backbone = SSLBackbone(model_name=ssl_model_name, pretrained=True, freeze=True)
        in_dim = getattr(self.ssl_backbone, "output_dim", 768)
        self.ssl_adapter = SSLAdapter(in_dim=in_dim, out_dim=clip_hidden, num_tokens=num_ssl_tokens)

        self.learning_rate = learning_rate
        self.diffusion_steps = diffusion_steps
        self.guidance_scale = guidance_scale
        self.init_strength = init_strength
        self.lambda_align = lambda_align
        self.image_log_interval = image_log_interval
        self.val_num_steps = val_num_steps
        self.val_guidance_scale = val_guidance_scale

        # scaling factor for SD VAE
        self.scaling = getattr(self.vae.config, "scaling_factor", 0.18215)

    def configure_optimizers(self):
        # Train adapter only
        trainable = list(self.ssl_adapter.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _encode_ssl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out_g = self.ssl_backbone(x)
            z_G = out_g["global"]   # (B, Dssl)
        ssl_tokens = self.ssl_adapter(z_G)       # (B, T, Cclip)
        return z_G.detach(), ssl_tokens

    def _text_embeds(self, batch_size: int, prompt_list: Optional[List[str]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncond_inputs = self.tokenizer(["" for _ in range(batch_size)], padding="max_length",
                                       max_length=self.tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt")
        uncond = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]
        cond: Optional[torch.Tensor] = None
        if prompt_list is not None and len(prompt_list) > 0:
            text_inputs = self.tokenizer(prompt_list, padding="max_length",
                                         max_length=self.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
            cond = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return uncond, cond

    def training_step(self, batch, batch_idx: int):
        degraded, clean = batch

        # SSL embeddings
        zG_clean, ssl_tokens_clean = self._encode_ssl(clean)       # stop-grad
        zD_deg,   ssl_tokens_deg   = self._encode_ssl(degraded)    # stop-grad on backbone, not on adapter

        # Build context (prepend SSL tokens, no text cond during training)
        bsz = degraded.size(0)
        uncond, _ = self._text_embeds(bsz, prompt_list=None)
        ctx = torch.cat([ssl_tokens_deg, uncond], dim=1)

        # VAE encode degraded -> z
        posterior = self.vae.encode(degraded).latent_dist
        z_unscaled = posterior.sample()
        z = z_unscaled * self.scaling

        # Single timestep training (faster, standard)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        z_t = self.scheduler.add_noise(z, noise, t)
        if hasattr(self.scheduler, "scale_model_input"):
            z_t = self.scheduler.scale_model_input(z_t, t)
        # ensure dtype consistency for AMP
        z_t = z_t.to(self.unet.dtype)
        ctx = ctx.to(self.unet.dtype)
        eps = self.unet(z_t, t, encoder_hidden_states=ctx).sample

        # Loss 1: standard denoise loss
        loss_denoise = F.mse_loss(eps, noise.detach())

        # Loss 2: alignment on SSL tokens via adapter projection (plain MSE)
        proj_clean = self.ssl_adapter(zG_clean)   # (B, T, C)
        proj_deg   = self.ssl_adapter(zD_deg)     # (B, T, C)
        loss_align = F.mse_loss(proj_deg, proj_clean.detach())

        loss = loss_denoise + self.lambda_align * loss_align
        self.log_dict({
            "train/loss": loss,
            "train/loss_denoise": loss_denoise,
            "train/loss_align": loss_align,
        }, prog_bar=True)
        return loss

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _psnr(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((x01 - y01) ** 2)
        eps = 1e-8
        return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

    @torch.no_grad()
    def _ddim_img2img(self, x_in: torch.Tensor, ssl_tokens: torch.Tensor, num_steps: int, strength: float, guidance_scale: float) -> torch.Tensor:
        device = x_in.device
        posterior = self.vae.encode(x_in).latent_dist
        z_unscaled = posterior.sample()
        z = z_unscaled * self.scaling

        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        t_start = timesteps[int(len(timesteps) * strength)]
        noise = torch.randn_like(z)
        z_t = self.scheduler.add_noise(z, noise, t_start)

        # text context: empty prompt
        uncond_inputs = self.tokenizer(["" for _ in range(x_in.size(0))], padding="max_length",
                                       max_length=self.tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt")
        uncond = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
        ctx_uncond = torch.cat([ssl_tokens, uncond], dim=1)

        for t in timesteps:
            if t > t_start:
                continue
            z_in = z_t
            if hasattr(self.scheduler, "scale_model_input"):
                z_in = self.scheduler.scale_model_input(z_in, t)
            # dtype match
            z_in = z_in.to(self.unet.dtype)
            ctx_un = ctx_uncond.to(self.unet.dtype)
            eps = self.unet(z_in, t, encoder_hidden_states=ctx_un).sample
            z_t = self.scheduler.step(eps, t, z_t).prev_sample

        x_rec = self.vae.decode(z_t / self.scaling).sample
        return x_rec

    def validation_step(self, batch, batch_idx: int):
        degraded, clean = batch

        # SSL embeddings
        zG_clean, ssl_tokens_clean = self._encode_ssl(clean)
        zD_deg, ssl_tokens_deg = self._encode_ssl(degraded)

        # Build context for denoise loss
        bsz = degraded.size(0)
        uncond, _ = self._text_embeds(bsz, prompt_list=None)
        ctx = torch.cat([ssl_tokens_deg, uncond], dim=1)

        # Denoise loss at one random timestep
        posterior = self.vae.encode(degraded).latent_dist
        z_unscaled = posterior.sample()
        z = z_unscaled * self.scaling
        noise = torch.randn_like(z)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        z_t = self.scheduler.add_noise(z, noise, t)
        if hasattr(self.scheduler, "scale_model_input"):
            z_t = self.scheduler.scale_model_input(z_t, t)
        z_t = z_t.to(self.unet.dtype)
        ctx = ctx.to(self.unet.dtype)
        eps = self.unet(z_t, t, encoder_hidden_states=ctx).sample
        loss_denoise = F.mse_loss(eps, noise)

        # Alignment loss on adapter tokens (plain MSE)
        proj_clean = self.ssl_adapter(zG_clean)
        proj_deg = self.ssl_adapter(zD_deg)
        loss_align = F.mse_loss(proj_deg, proj_clean.detach())

        self.log("val/loss_denoise", loss_denoise, prog_bar=True)
        self.log("val/loss_align", loss_align, prog_bar=True)

        # Direct decode PSNR
        with torch.no_grad():
            direct_rec = self.vae.decode(z_unscaled).sample
            psnr_direct = self._psnr(self._to_01(direct_rec), self._to_01(clean))
            self.log("val/psnr_direct", psnr_direct, prog_bar=True)

        # Periodic DDIM sampling and image dump
        if (self.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):
            with torch.no_grad():
                recon = self._ddim_img2img(
                    degraded,
                    ssl_tokens=self.ssl_adapter(zD_deg),
                    num_steps=self.val_num_steps,
                    strength=self.init_strength,
                    guidance_scale=self.val_guidance_scale,
                )
                psnr_diff = self._psnr(self._to_01(recon), self._to_01(clean))
                self.log("val/psnr_diffusion", psnr_diff, prog_bar=True)

                # Save comparison grid
                n = min(4, degraded.size(0))
                out_dir = os.path.join("validation_samples_ssl2", f"epoch_{self.current_epoch}")
                os.makedirs(out_dir, exist_ok=True)
                random_idx = torch.randint(0, degraded.size(0), (n,))
                tiles = []
                for i in range(n):
                    idx = random_idx[i]
                    di = self._to_01(degraded[idx])      # (3,H,W)
                    ci = self._to_01(clean[idx])         # (3,H,W)
                    dr = self._to_01(direct_rec[idx])    # (3,H,W)
                    rr = self._to_01(recon[idx])         # (3,H,W)
                    tiles.extend([di, ci, dr, rr])
                save_image(tiles, os.path.join(out_dir, "comparison.png"), nrow=4, normalize=False, padding=2)
        return {"val_loss": loss_denoise + self.lambda_align * loss_align}


class SSLDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_root: str, 
        batch_size: int, 
        num_workers: int = 4, 
        crop_size: int = 256, 
        use_sd_scaling: bool = True,
        use_multi_degradation: bool = True,
        degradation_types: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = (crop_size, crop_size)
        self.use_sd_scaling = use_sd_scaling
        self.use_multi_degradation = use_multi_degradation
        self.degradation_types = degradation_types or ['denoise_15', 'dehaze', 'deblur', 'lowlight']

        if use_sd_scaling:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])
        else:
            self.image_transform = transforms.ToTensor()

    def setup(self, stage=None):
        if self.use_multi_degradation:
            # Use multi-degradation dataset
            self.train_dataset = create_multi_degradation_dataset(
                data_root=self.data_root,
                patch_size=self.crop_size[0],
                degradation_types=self.degradation_types,
            )
            
            # Create val dataset with center crop and smaller subset
            val_dataset_full = create_multi_degradation_dataset(
                data_root=self.data_root,
                patch_size=self.crop_size[0],
                degradation_types=self.degradation_types,
            )
            
            # Use subset for faster validation
            val_size = min(len(val_dataset_full), 500)
            val_indices = list(range(val_size))
            self.val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
            
        else:
            # Use original paired dataset
            base_full = DegradedCleanPairDataset(self.data_root)
            train_size = int(0.95 * len(base_full))
            val_size = len(base_full) - train_size
            temp_train, temp_val = torch.utils.data.random_split(base_full, [train_size, val_size])
            train_indices = temp_train.indices
            val_indices = temp_val.indices

            train_base = DegradedCleanPairDataset(
                data_root=self.data_root,
                paired_transform=PairedRandomCropFlip(crop_size=self.crop_size),
                image_transform=self.image_transform,
            )
            val_base = DegradedCleanPairDataset(
                data_root=self.data_root,
                paired_transform=PairedCenterCrop(crop_size=self.crop_size),
                image_transform=self.image_transform,
            )

            self.train_dataset = torch.utils.data.Subset(train_base, train_indices)
            self.val_dataset = torch.utils.data.Subset(val_base, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)


