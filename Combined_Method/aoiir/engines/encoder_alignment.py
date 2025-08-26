import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

from aoiir.models.compvis_vqvae import CompVisVQVAE
from aoiir.models.promptir import PromptAwareEncoder, PromptGenModule
from aoiir.datasets.paired import (
    DegradedCleanPairDataset,
    PairedRandomCropFlip,
    PairedCenterCrop,
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torchvision.models as models

def coral_map(z):  # z: [B,C,H,W]
    B,C,H,W = z.shape
    X = z.permute(0,2,3,1).reshape(-1, C)     # [N, C], N=B*H*W
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.t() @ X) / (X.size(0)-1)         # [C, C]
    return cov

def ssim_loss(img1, img2):
    """SSIM loss function (1 - SSIM)"""
    # Convert to [0, 1] range if inputs are in [-1, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0
    
    # Clamp to ensure values are in [0, 1]
    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)
    
    ssim_val = ssim(img1, img2, data_range=1.0)
    return 1.0 - ssim_val

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.features = nn.ModuleList()
        
        prev_layer = 0
        for layer in layers:
            self.features.append(vgg[prev_layer:layer])
            prev_layer = layer
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # Convert to [0, 1] range if inputs are in [-1, 1] range
        if x.min() < 0:
            x = (x + 1.0) / 2.0
            y = (y + 1.0) / 2.0
        
        # Clamp to ensure values are in [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        for feature_extractor in self.features:
            x = feature_extractor(x)
            y = feature_extractor(y)
            loss += F.mse_loss(x, y)
        
        return loss

# Global instance for LPIPS loss
_lpips_loss_fn = None

def lpips_loss(img1, img2):
    """LPIPS-style perceptual loss using VGG features"""
    global _lpips_loss_fn
    if _lpips_loss_fn is None:
        _lpips_loss_fn = VGGPerceptualLoss()
        _lpips_loss_fn.eval()
    
    if img1.device != next(_lpips_loss_fn.parameters()).device:
        _lpips_loss_fn = _lpips_loss_fn.to(img1.device)
    
    return _lpips_loss_fn(img1, img2)


def freeze(model):
    if hasattr(model, 'parameters'):
        for p in model.parameters():
            p.requires_grad = False

class EncoderAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_autoencoder_path,
        learning_rate=1e-4,
        similarity_weight=1.0,
        l2_weight=1.0,
        img_loss_weight=0.5,
        use_stable_diffusion=False,
        use_compvis_vqvae=False,
        diffusion_steps=1000,
        init_strength=0.2,
        guidance_scale=1.0,
        compvis_model_path="model.ckpt",
        sd_model_name="stabilityai/stable-diffusion-2-1-base",
        enable_promptir=True,
        image_log_interval=10,
        color_correction_on=True,
        supir_sampling_enabled=False,
        supir_noise_level=0.1,
        supir_restoration_guidance=1.5,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.similarity_weight = similarity_weight
        self.l2_weight = l2_weight
        self.img_loss_weight = img_loss_weight
        self.use_stable_diffusion = use_stable_diffusion
        self.use_compvis_vqvae = use_compvis_vqvae
        self.compvis_model_path = compvis_model_path
        self.sd_model_name = sd_model_name
        self.enable_promptir = enable_promptir
        self.image_log_interval = image_log_interval
        self.diffusion_steps = diffusion_steps
        self.init_strength = init_strength
        self.guidance_scale = guidance_scale
        self.color_correction_on = color_correction_on
        # SUPIR sampling parameters
        self.supir_sampling_enabled = supir_sampling_enabled
        self.supir_noise_level = supir_noise_level
        self.supir_restoration_guidance = supir_restoration_guidance
        self.supir_s_churn = 0.0
        self.supir_s_tmin = 0.0
        self.supir_s_tmax = float('inf')
        self.supir_s_noise = 1.003
        # diffusion disabled in validation

        if use_compvis_vqvae:
            self.autoencoder = CompVisVQVAE(model_path=compvis_model_path)
        elif use_stable_diffusion:
            assert AutoencoderKL is not None, "diffusers not installed"
            self.autoencoder = AutoencoderKL.from_pretrained(
                self.sd_model_name, subfolder="vae", torch_dtype=torch.float32
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.sd_model_name, subfolder="text_encoder", torch_dtype=torch.float32
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.sd_model_name, subfolder="tokenizer"
            )
            self.scheduler = DDIMScheduler.from_pretrained(
                self.sd_model_name, subfolder="scheduler"
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                self.sd_model_name, subfolder="unet", torch_dtype=torch.float32
            ).eval()

            freeze(self.text_encoder)
            freeze(self.tokenizer)
            freeze(self.scheduler)
            freeze(self.unet)
        else:
            from taming.models.vqgan import Autoencoder as TamingAutoencoder

            self.autoencoder = TamingAutoencoder.load_from_checkpoint(pretrained_autoencoder_path)

        freeze(self.autoencoder)

        if use_compvis_vqvae:
            import copy

            self.ae_model = self.autoencoder.model
            self.encoder_degraded = copy.deepcopy(self.ae_model.encoder)
            self.quant_conv_degraded = copy.deepcopy(self.ae_model.quant_conv)
            self.quantize_shared = self.ae_model.quantize
            self.prompt_gen_module = PromptGenModule(prompt_dim=128, num_prompts=5)
        elif use_stable_diffusion:
            temp_vae = AutoencoderKL.from_pretrained(
                self.sd_model_name, subfolder="vae", torch_dtype=torch.float32
            )
            base_encoder = temp_vae.encoder
            base_encoder.load_state_dict(self.autoencoder.encoder.state_dict())
            if self.enable_promptir:
                self.encoder_degraded = PromptAwareEncoder(base_encoder=base_encoder, prompt_dim=128, num_prompts=5)
            else:
                self.encoder_degraded = base_encoder
            del temp_vae
        else:
            base_encoder = type(self.autoencoder.encoder)(
                **{k: v for k, v in self.autoencoder.encoder.__dict__.items() if not k.startswith('_')}
            )
            base_encoder.load_state_dict(self.autoencoder.encoder.state_dict())
            if self.enable_promptir:
                self.encoder_degraded = PromptAwareEncoder(base_encoder=base_encoder, prompt_dim=128, num_prompts=5)
            else:
                self.encoder_degraded = base_encoder

        if use_compvis_vqvae:
            for p in self.encoder_degraded.parameters():
                p.requires_grad = True
            for p in self.quant_conv_degraded.parameters():
                p.requires_grad = True
            for p in self.prompt_gen_module.parameters():
                p.requires_grad = True
        else:
            for p in self.encoder_degraded.parameters():
                p.requires_grad = True

    # diffusion helpers removed

    @staticmethod
    def _to_unet_channels(z3: torch.Tensor) -> torch.Tensor:
        if z3.shape[1] == 4:
            return z3
        pad = (0, 0, 0, 0, 0, 1)
        return torch.nn.functional.pad(z3, pad)

    @staticmethod
    def _from_unet_channels(z4: torch.Tensor) -> torch.Tensor:
        if z4.shape[1] >= 3:
            return z4[:, :3]
        raise ValueError("UNet output has <3 channels")

    # diffusion helpers removed

    # diffusion helpers removed

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _psnr(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        # x01, y01 in [0,1]
        mse = torch.mean((x01 - y01) ** 2)
        eps = 1e-8
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
        return psnr

    def forward(self, degraded_img, clean_img):
        if self.use_compvis_vqvae:
            _h_D = self.encoder_degraded(degraded_img)
            _h_D = self.quant_conv_degraded(_h_D)
            z_D, _, _ = self.quantize_shared(_h_D)

            _h_G = self.autoencoder.model.encoder(clean_img)
            _h_G = self.autoencoder.model.quant_conv(_h_G)
            z_G, _, _ = self.autoencoder.model.quantize(_h_G)
        elif self.use_stable_diffusion:
            encoder_output_D = self.encoder_degraded(degraded_img)
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

            posterior_D = DiagonalGaussianDistribution(encoder_output_D)
            z_D = posterior_D.sample()
            z_G = self.autoencoder.encode(clean_img).latent_dist.sample()
        else:
            z_D = self.encoder_degraded(degraded_img)
            z_G = self.autoencoder.encoder(clean_img)

        return z_D, z_G

    def training_step(self, batch, batch_idx):
        degraded_imgs, clean_imgs = batch
        z_D, z_G = self(degraded_imgs, clean_imgs)

        similarity = F.cosine_similarity(z_D, z_G, dim=1).mean()
        l2_loss = F.mse_loss(z_D, z_G)
        
        # Decode to get reconstructed images for image-level loss
        if self.use_stable_diffusion:
            D_out = self.autoencoder.decode(z_D).sample
            G_out = self.autoencoder.decode(z_G).sample
        elif self.use_compvis_vqvae:
            D_out = self.autoencoder.decode(z_D)
            G_out = self.autoencoder.decode(z_G)
        else:
            D_out = self.autoencoder.decoder(z_D)
            G_out = self.autoencoder.decoder(z_G)
        
        # Calculate image-level losses
        D_l2_loss = F.mse_loss(D_out, G_out)
        img_loss = D_l2_loss

        # Combined loss
        loss = (-self.similarity_weight * similarity + 
                self.l2_weight * l2_loss + 
                self.img_loss_weight * img_loss)
        
        # Logging
        self.log('train/similarity', similarity, prog_bar=True)
        self.log('train/l2_loss', l2_loss, prog_bar=True)
        self.log('train/img_loss', img_loss, prog_bar=True)
        self.log('train/D_l2_loss', D_l2_loss)
        self.log('train/total_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Support dict of loaders from per-degradation val
        if isinstance(batch, dict):
            # Lightning won't pass dict here; kept for safety
            return None
        degraded_imgs, clean_imgs = batch
        if (self.trainer.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):
            try:
                self._save_input_debug(degraded_imgs, clean_imgs)
            except Exception:
                pass
        z_D, z_G = self(degraded_imgs, clean_imgs)
        l2_loss = F.mse_loss(z_D, z_G)
        similarity = F.cosine_similarity(z_D, z_G, dim=1).mean()
        self.log('val/similarity', similarity, prog_bar=True)
        self.log('val/l2_loss', l2_loss, prog_bar=True)

        # PSNR and SSIM without diffusion (direct decode)
        with torch.no_grad():
            if self.use_stable_diffusion:
                restored_direct_D = self.autoencoder.decode(z_D).sample
                restored_direct_G = self.autoencoder.decode(z_G).sample
            elif self.use_compvis_vqvae:
                restored_direct_D = self.autoencoder.decode(z_D)
                restored_direct_G = self.autoencoder.decode(z_G)
            else:
                restored_direct_D = self.autoencoder.decoder(z_D)
                restored_direct_G = self.autoencoder.decoder(z_G)
            psnr_direct = self._psnr(self._to_01(restored_direct_D), self._to_01(clean_imgs))
            self.log('val/psnr_direct', psnr_direct, prog_bar=True)
            ssim_direct = self._ssim(self._to_01(restored_direct_D), self._to_01(clean_imgs))
            self.log('val/ssim_direct', ssim_direct, prog_bar=True)
            if (self.trainer.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):    
                if self.use_stable_diffusion:
                    # diffusion on both degraded and clean latents
                    if self.supir_sampling_enabled:
                        # Use SUPIR-style sampling for better restoration quality
                        restored_diffusion_D = self.sd_latent_supir_sampling(
                            z_D, prompt="", 
                            restoration_guidance=self.supir_restoration_guidance,
                            noise_level=self.supir_noise_level
                        )
                        restored_diffusion_G = self.sd_latent_supir_sampling(
                            z_G, prompt="", 
                            restoration_guidance=self.supir_restoration_guidance,
                            noise_level=self.supir_noise_level
                        )
                    else:
                        # Use standard img2img sampling
                        restored_diffusion_D = self.sd_latent_img2img(z_D, prompt="")
                        restored_diffusion_G = self.sd_latent_img2img(z_G, prompt="")
                    
                    if self.color_correction_on:
                        # Color correction is already applied in SUPIR sampling if enabled
                        restored_diffusion_D = self.color_correction(restored_diffusion_D, degraded_imgs)
                    psnr_diffusion = self._psnr(self._to_01(restored_diffusion_D), self._to_01(clean_imgs))
                    psnr_key = 'val/psnr_supir' if self.supir_sampling_enabled else 'val/psnr_diffusion'
                    self.log(psnr_key, psnr_diffusion, prog_bar=True)
                    ssim_diffusion = self._ssim(self._to_01(restored_diffusion_D), self._to_01(clean_imgs))
                    ssim_key = 'val/ssim_supir' if self.supir_sampling_enabled else 'val/ssim_diffusion'
                    self.log(ssim_key, ssim_diffusion, prog_bar=True)
                    
                    # Log which sampling method is being used
                    if self.supir_sampling_enabled:
                        self.log('val/supir_noise_level', self.supir_noise_level)
                        self.log('val/supir_restoration_guidance', self.supir_restoration_guidance)
                    # Score refinement (optional visualization)
                    # restored_score_D = self.sd_latent_score_refine(z_D, prompt="", t_frac=0.5, refine_steps=10)
                    # restored_score_G = self.sd_latent_score_refine(z_G, prompt="", t_frac=0.5, refine_steps=10)
                    self._generate_restoration_samples_both_paths(
                        degraded_imgs, clean_imgs,
                        restored_direct_D, restored_diffusion_D,
                        restored_direct_G, restored_diffusion_G,
                    )
                    # self._generate_restoration_samples_score(
                    #     degraded_imgs, clean_imgs,
                    #     restored_direct_D, restored_score_D,
                    #     restored_direct_G, restored_score_G,
                    # )

        return similarity

    def color_correction(self, restored_img, degraded_img):
        # restored_img and degraded_img : [B, 3, H, W]
        xr_mean = restored_img.mean(dim=(2, 3), keepdim=True)
        xr_std = restored_img.std(dim=(2, 3), keepdim=True)
        xd_mean = degraded_img.mean(dim=(2, 3), keepdim=True)
        xd_std = degraded_img.std(dim=(2, 3), keepdim=True)

        return (restored_img - xr_mean) * (xd_std / xr_std) + xd_mean

    def _save_input_debug(self, degraded_imgs, clean_imgs):
        from torchvision.utils import save_image
        n_samples = min(4, degraded_imgs.size(0))
        if self.use_stable_diffusion or self.use_compvis_vqvae:
            degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
            clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
        else:
            degraded_display = torch.clamp((degraded_imgs[:n_samples] / 0.1) + 0.5, 0, 1)
            clean_display = torch.clamp((clean_imgs[:n_samples] / 0.1) + 0.5, 0, 1)

        grid = []
        for i in range(n_samples):
            grid.extend([degraded_display[i], clean_display[i]])

        os.makedirs(f"validation_samples/epoch_{self.trainer.current_epoch}", exist_ok=True)
        save_image(
            grid,
            f"validation_samples/epoch_{self.trainer.current_epoch}/inputs_degraded_vs_clean.png",
            nrow=2,
            normalize=False,
            padding=2,
        )

    def _generate_restoration_samples_both_paths(self, degraded_imgs, clean_imgs,
                                                 restored_direct_D, restored_diffusion_D,
                                                 restored_direct_G, restored_diffusion_G):
        try:
            with torch.no_grad():
                n_samples = min(4, degraded_imgs.size(0))
                degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_direct_D_display = torch.clamp((restored_direct_D[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_diffusion_D_display = torch.clamp((restored_diffusion_D[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_direct_G_display = torch.clamp((restored_direct_G[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_diffusion_G_display = torch.clamp((restored_diffusion_G[:n_samples] + 1.0) / 2.0, 0, 1)
                
                comparison_imgs = []
                for i in range(n_samples):
                    target_h, target_w = clean_display[i].shape[-2], clean_display[i].shape[-1]

                    def resize_to(img, h, w):
                        if img.shape[-2] == h and img.shape[-1] == w:
                            return img
                        return F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

                    di = resize_to(degraded_display[i], target_h, target_w)
                    ci = resize_to(clean_display[i], target_h, target_w)
                    d_dir = resize_to(restored_direct_D_display[i], target_h, target_w)
                    d_dif = resize_to(restored_diffusion_D_display[i], target_h, target_w)
                    g_dir = resize_to(restored_direct_G_display[i], target_h, target_w)
                    g_dif = resize_to(restored_diffusion_G_display[i], target_h, target_w)

                    # layout: degraded | clean | D_direct | D_diff | G_direct | G_diff
                    comparison_imgs.extend([di, ci, d_dir, d_dif, g_dir, g_dif])

                output_dir = f"validation_samples/epoch_{self.trainer.current_epoch}"
                os.makedirs(output_dir, exist_ok=True)
                filename = "restoration_comparison_SUPIR.png" if self.supir_sampling_enabled else "restoration_comparison_inc_diffusion.png"
                save_image(comparison_imgs, f"{output_dir}/{filename}", nrow=6, normalize=False, padding=2)

        except Exception as e:
            print(f"Error generating restoration samples with diffusion: {e}")
                


    def _generate_restoration_samples_score(self, degraded_imgs, clean_imgs,
                                            restored_direct_D, restored_score_D,
                                            restored_direct_G, restored_score_G):
        try:
            with torch.no_grad():
                n_samples = min(4, degraded_imgs.size(0))
                degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                rdir_D = torch.clamp((restored_direct_D[:n_samples] + 1.0) / 2.0, 0, 1)
                rsc_D = torch.clamp((restored_score_D[:n_samples] + 1.0) / 2.0, 0, 1)
                rdir_G = torch.clamp((restored_direct_G[:n_samples] + 1.0) / 2.0, 0, 1)
                rsc_G = torch.clamp((restored_score_G[:n_samples] + 1.0) / 2.0, 0, 1)

                comparison_imgs = []
                for i in range(n_samples):
                    target_h, target_w = clean_display[i].shape[-2], clean_display[i].shape[-1]

                    def resize_to(img, h, w):
                        if img.shape[-2] == h and img.shape[-1] == w:
                            return img
                        return F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

                    di = resize_to(degraded_display[i], target_h, target_w)
                    ci = resize_to(clean_display[i], target_h, target_w)
                    d_dir = resize_to(rdir_D[i], target_h, target_w)
                    d_sc = resize_to(rsc_D[i], target_h, target_w)
                    g_dir = resize_to(rdir_G[i], target_h, target_w)
                    g_sc = resize_to(rsc_G[i], target_h, target_w)

                    # layout: degraded | clean | D_direct | D_score | G_direct | G_score
                    comparison_imgs.extend([di, ci, d_dir, d_sc, g_dir, g_sc])

                output_dir = f"validation_samples/epoch_{self.trainer.current_epoch}"
                os.makedirs(output_dir, exist_ok=True)
                save_image(comparison_imgs, f"{output_dir}/restoration_comparison_score_refine.png", nrow=6, normalize=False, padding=2)

        except Exception as e:
            print(f"Error generating restoration samples (score refine): {e}")

    @torch.no_grad()
    def sd_latent_img2img(self, z_unscaled, prompt=""):
        """DDIM img2img with proper scaling and resizing for SD UNet compatibility."""
        device = z_unscaled.device
        batch_size = z_unscaled.shape[0]
        
        # 1) Get scaling factor correctly
        scaling = getattr(self.autoencoder.config, "scaling_factor", 0.18215)
        z = z_unscaled * scaling
        
        # 2) Handle UNet expected sample_size (e.g., 64 for SD 512)
        orig_h, orig_w = z.shape[-2], z.shape[-1]
        target_size = getattr(self.unet.config, "sample_size", 64)
        need_resize = (orig_h != target_size or orig_w != target_size)
        need_resize = False
        if need_resize:
            z_work = F.interpolate(z, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            z_work = z
        
        # 3) Text embeddings (CFG)
        uncond_inputs = self.tokenizer(["" for _ in range(batch_size)], padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
        uncond = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
        
        cond = None
        if prompt != "":  # Fix: use != instead of 'is not'
            text_inputs = self.tokenizer([prompt for _ in range(batch_size)], padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            cond = self.text_encoder(text_inputs.input_ids.to(device))[0]
        
        # 4) DDIM setup (img2img: add noise then denoise)
        self.scheduler.set_timesteps(self.diffusion_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Fix: proper strength handling (1.0 = full denoise, 0.0 = no denoise)
        t_start_idx = int(len(timesteps) * self.init_strength)
        t_start = timesteps[t_start_idx] if t_start_idx < len(timesteps) else timesteps[0]

        noise = torch.randn_like(z_work)
        z_t = self.scheduler.add_noise(z_work, noise, t_start)
        
        # 5) Denoise loop
        for t in timesteps:
            if t > t_start:
                continue
                
            if cond is not None and self.guidance_scale > 1.0:
                # Classifier-free guidance
                z_in = torch.cat([z_t, z_t], dim=0)
                z_in = self.scheduler.scale_model_input(z_in, t)
                cond_in = torch.cat([uncond, cond], dim=0)
                noise_pred = self.unet(z_in, t, encoder_hidden_states=cond_in).sample
                noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            else:
                # No guidance
                z_t = self.scheduler.scale_model_input(z_t, t)
                noise_pred = self.unet(z_t, t, encoder_hidden_states=uncond).sample
                
            z_t = self.scheduler.step(noise_pred, t, z_t).prev_sample
        
        # 6) Resize back if needed
        if need_resize:
            z_out = F.interpolate(z_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        else:
            z_out = z_t
            
        z_unscaled_out = z_out / scaling
        x_out = self.autoencoder.decode(z_unscaled_out).sample
        return x_out

    @torch.no_grad()
    def sd_latent_supir_sampling(self, z_unscaled, prompt="", restoration_guidance=1.5, noise_level=None):
        """SUPIR-style sampling with EDM sampler and restoration-specific guidance.
        Based on the SUPIR paper: https://arxiv.org/pdf/2401.13627
        """
        assert self.use_stable_diffusion, "SUPIR sampling only supported for Stable Diffusion path"
        device = z_unscaled.device
        scaling = getattr(self.autoencoder.config, "scaling_factor", 0.18215)
        z0 = z_unscaled * scaling
        batch_size, _, orig_h, orig_w = z0.shape
        
        # Handle UNet expected sample_size
        target_size = getattr(self.unet.config, "sample_size", 64)
        need_resize = (orig_h != target_size or orig_w != target_size)
        if need_resize:
            z_work = F.interpolate(z0, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            z_work = z0
        
        # Text embeddings for restoration guidance
        with torch.no_grad():
            # Negative prompt for degraded images
            negative_prompts = ["blurry, noise, artifacts, low quality" for _ in range(batch_size)]
            negative_inputs = self.tokenizer(negative_prompts, padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt")
            negative_embed = self.text_encoder(negative_inputs.input_ids.to(device))[0]
            
            # Positive prompt for high-quality restoration
            if prompt == "":
                prompt = "high quality, sharp, detailed, restored image"
            positive_prompts = [prompt for _ in range(batch_size)]
            positive_inputs = self.tokenizer(positive_prompts, padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt")
            positive_embed = self.text_encoder(positive_inputs.input_ids.to(device))[0]
        
        # SUPIR-style diffusion with EDM sampler
        self.scheduler.set_timesteps(self.diffusion_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Initialize with controlled noise injection (SUPIR technique)
        noise_level = noise_level if noise_level is not None else self.supir_noise_level
        noise = torch.randn_like(z_work) * noise_level
        z_t = z_work + noise  # Direct noise addition instead of scheduler noise
        
        # Get starting timestep based on noise level
        t_start_idx = int(len(timesteps) * (1.0 - noise_level))
        t_start_idx = max(0, min(t_start_idx, len(timesteps) - 1))
        
        # EDM-style sampling loop with restoration guidance
        for i, t in enumerate(timesteps[t_start_idx:]):
            # Stochastic noise injection (EDM/SUPIR technique)
            if self.supir_s_tmin <= t <= self.supir_s_tmax and self.supir_s_noise > 0:
                gamma = min(self.supir_s_churn / (len(timesteps) - t_start_idx), 2**0.5 - 1)
                if gamma > 0:
                    eps = torch.randn_like(z_t) * self.supir_s_noise
                    z_t = z_t + eps * (gamma * ((t / 1000) ** 2)).sqrt()
            
            # Classifier-free guidance with restoration-specific conditioning
            z_in = torch.cat([z_t, z_t, z_t], dim=0)
            cond_in = torch.cat([negative_embed, positive_embed, positive_embed], dim=0)
            
            # Predict noise with three conditions
            noise_pred = self.unet(z_in, t, encoder_hidden_states=cond_in).sample
            noise_negative, noise_positive, noise_positive2 = noise_pred.chunk(3, dim=0)
            
            # SUPIR-style guidance combination
            noise_pred = (noise_negative + 
                         restoration_guidance * (noise_positive - noise_negative) +
                         0.5 * (noise_positive2 - noise_positive))  # Additional refinement
            
            # Denoise step
            z_t = self.scheduler.step(noise_pred, t, z_t).prev_sample
            
            # Optional: Progressive sharpening (SUPIR technique)
            if i % 5 == 0 and i > 0:
                # Add small high-frequency details
                high_freq = z_t - F.avg_pool2d(F.avg_pool2d(z_t, 3, 1, 1), 3, 1, 1)
                z_t = z_t + 0.1 * high_freq
        
        # Resize back if needed
        if need_resize:
            z_out = F.interpolate(z_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        else:
            z_out = z_t
        
        z_unscaled_out = z_out / scaling
        x_out = self.autoencoder.decode(z_unscaled_out).sample
        
        return x_out

    def sd_latent_score_refine(self, z_unscaled, prompt="", t_frac: float = 0.5, refine_steps: int = 10, eta: float = 0.1):
        """Mode-seeking score refinement at a fixed timestep using ε-parameterization.
        Uses s(x_t,t) ≈ -ε_θ(x_t,t)/σ_t, updates z_t ← z_t - η·ε/σ, then projects to x0 and decodes.
        """
        assert self.use_stable_diffusion, "score refinement only supported for Stable Diffusion path"
        device = z_unscaled.device
        scaling = getattr(self.autoencoder, "scaling_factor", 0.18215)
        z0 = z_unscaled * scaling
        bsz, _, orig_h, orig_w = z0.shape

        # Handle UNet expected sample_size (e.g., 64 for SD 512)
        target_size = getattr(self.unet.config, "sample_size", orig_h)
        need_resize = (orig_h != target_size or orig_w != target_size)
        if need_resize:
            z0_work = F.interpolate(z0, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            z0_work = z0

        # Conditioning
        with torch.no_grad():
            uncond_inputs = self.tokenizer(["" for _ in range(bsz)], padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt")
            uncond = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
            cond = None
            if prompt:
                text_inputs = self.tokenizer([prompt for _ in range(bsz)], padding="max_length",
                                             max_length=self.tokenizer.model_max_length,
                                             truncation=True, return_tensors="pt")
                cond = self.text_encoder(text_inputs.input_ids.to(device))[0]

        # Fixed timestep t and init z_t via img2img noise
        self.scheduler.set_timesteps(self.diffusion_steps, device=device)
        timesteps = self.scheduler.timesteps
        t_idx = int(len(timesteps) * float(t_frac))
        t = timesteps[t_idx]
        with torch.no_grad():
            noise = torch.randn_like(z0_work)
            z_t = self.scheduler.add_noise(z0_work, noise, t)

        # Precompute ᾱ_t and σ_t
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        alpha_bar = alphas_cumprod[t.long()]
        alpha_bar = alpha_bar.view(1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_bar)

        # Helper: predict eps with CFG
        def predict_eps(x):
            if cond is not None:
                x_in = torch.cat([x, x], dim=0)
                cond_in = torch.cat([uncond, cond], dim=0)
                eps = self.unet(x_in, t, encoder_hidden_states=cond_in).sample
                eps_u, eps_c = eps.chunk(2, dim=0)
                return eps_u + self.guidance_scale * (eps_c - eps_u)
            else:
                return self.unet(x, t, encoder_hidden_states=uncond).sample

        # Score-ascent updates at fixed t: z_t ← z_t - η·ε/σ
        with torch.no_grad():
            for _ in range(int(refine_steps)):
                eps_hat = predict_eps(z_t)
                z_t = z_t - eta * eps_hat / sigma_t

            # Project to x0 via DDPM formula: x0 = (z_t - σ·ε)/√ᾱ
            eps_hat = predict_eps(z_t)
            z0_pred = (z_t - sigma_t * eps_hat) / torch.sqrt(alpha_bar)

            # Resize back if needed
            if need_resize:
                z0_pred = F.interpolate(z0_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            z_unscaled_out = z0_pred / scaling
            x_out = self.autoencoder.decode(z_unscaled_out).sample
            return x_out

    def _psnr(self, x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        return psnr(x01, y01, data_range=1.0)

    def _ssim(self, x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        return ssim(x01, y01, data_range=1.0)

    def configure_optimizers(self):
        if self.use_compvis_vqvae:
            trainable_params = list(self.encoder_degraded.parameters()) + list(self.quant_conv_degraded.parameters())
        else:
            trainable_params = self.encoder_degraded.parameters()
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/similarity"}}


class AlignmentDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, num_workers=0, train_split=0.95, use_compvis_vqvae=False, use_stable_diffusion=False, crop_size=256):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.use_compvis_vqvae = use_compvis_vqvae
        self.use_stable_diffusion = use_stable_diffusion
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        

        if self.use_compvis_vqvae or self.use_stable_diffusion:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def setup(self, stage=None):
        temp_full = DegradedCleanPairDataset(data_root=self.data_root, paired_transform=None, image_transform=None)
        train_size = int(self.train_split * len(temp_full))
        val_size = len(temp_full) - train_size
        temp_train, temp_val = torch.utils.data.random_split(temp_full, [train_size, val_size])
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

from aoiir.models.compvis_vqvae import CompVisVQVAE
from aoiir.models.promptir import PromptAwareEncoder, PromptGenModule
from aoiir.datasets.paired import (
    DegradedCleanPairDataset,
    PairedRandomCropFlip,
    PairedCenterCrop,
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torchvision.models as models

def coral_map(z):  # z: [B,C,H,W]
    B,C,H,W = z.shape
    X = z.permute(0,2,3,1).reshape(-1, C)     # [N, C], N=B*H*W
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.t() @ X) / (X.size(0)-1)         # [C, C]
    return cov

def ssim_loss(img1, img2):
    """SSIM loss function (1 - SSIM)"""
    # Convert to [0, 1] range if inputs are in [-1, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0
    
    # Clamp to ensure values are in [0, 1]
    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)
    
    ssim_val = ssim(img1, img2, data_range=1.0)
    return 1.0 - ssim_val

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.features = nn.ModuleList()
        
        prev_layer = 0
        for layer in layers:
            self.features.append(vgg[prev_layer:layer])
            prev_layer = layer
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # Convert to [0, 1] range if inputs are in [-1, 1] range
        if x.min() < 0:
            x = (x + 1.0) / 2.0
            y = (y + 1.0) / 2.0
        
        # Clamp to ensure values are in [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        for feature_extractor in self.features:
            x = feature_extractor(x)
            y = feature_extractor(y)
            loss += F.mse_loss(x, y)
        
        return loss

# Global instance for LPIPS loss
_lpips_loss_fn = None

def lpips_loss(img1, img2):
    """LPIPS-style perceptual loss using VGG features"""
    global _lpips_loss_fn
    if _lpips_loss_fn is None:
        _lpips_loss_fn = VGGPerceptualLoss()
        _lpips_loss_fn.eval()
    
    if img1.device != next(_lpips_loss_fn.parameters()).device:
        _lpips_loss_fn = _lpips_loss_fn.to(img1.device)
    
    return _lpips_loss_fn(img1, img2)


def freeze(model):
    if hasattr(model, 'parameters'):
        for p in model.parameters():
            p.requires_grad = False

class EncoderAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_autoencoder_path,
        learning_rate=1e-4,
        similarity_weight=1.0,
        l2_weight=1.0,
        img_loss_weight=0.5,
        use_stable_diffusion=False,
        use_compvis_vqvae=False,
        diffusion_steps=1000,
        init_strength=0.2,
        guidance_scale=1.0,
        compvis_model_path="model.ckpt",
        sd_model_name="stabilityai/stable-diffusion-2-1-base",
        enable_promptir=True,
        image_log_interval=10,
        color_correction_on=True,
        supir_sampling_enabled=False,
        supir_noise_level=0.1,
        supir_restoration_guidance=1.5,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.similarity_weight = similarity_weight
        self.l2_weight = l2_weight
        self.img_loss_weight = img_loss_weight
        self.use_stable_diffusion = use_stable_diffusion
        self.use_compvis_vqvae = use_compvis_vqvae
        self.compvis_model_path = compvis_model_path
        self.sd_model_name = sd_model_name
        self.enable_promptir = enable_promptir
        self.image_log_interval = image_log_interval
        self.diffusion_steps = diffusion_steps
        self.init_strength = init_strength
        self.guidance_scale = guidance_scale
        self.color_correction_on = color_correction_on
        # SUPIR sampling parameters
        self.supir_sampling_enabled = supir_sampling_enabled
        self.supir_noise_level = supir_noise_level
        self.supir_restoration_guidance = supir_restoration_guidance
        self.supir_s_churn = 0.0
        self.supir_s_tmin = 0.0
        self.supir_s_tmax = float('inf')
        self.supir_s_noise = 1.003
        # diffusion disabled in validation

        if use_compvis_vqvae:
            self.autoencoder = CompVisVQVAE(model_path=compvis_model_path)
        elif use_stable_diffusion:
            assert AutoencoderKL is not None, "diffusers not installed"
            self.autoencoder = AutoencoderKL.from_pretrained(
                self.sd_model_name, subfolder="vae", torch_dtype=torch.float32
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.sd_model_name, subfolder="text_encoder", torch_dtype=torch.float32
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.sd_model_name, subfolder="tokenizer"
            )
            self.scheduler = DDIMScheduler.from_pretrained(
                self.sd_model_name, subfolder="scheduler"
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                self.sd_model_name, subfolder="unet", torch_dtype=torch.float32
            ).eval()

            freeze(self.text_encoder)
            freeze(self.tokenizer)
            freeze(self.scheduler)
            freeze(self.unet)
        else:
            from taming.models.vqgan import Autoencoder as TamingAutoencoder

            self.autoencoder = TamingAutoencoder.load_from_checkpoint(pretrained_autoencoder_path)

        freeze(self.autoencoder)

        if use_compvis_vqvae:
            import copy

            self.ae_model = self.autoencoder.model
            self.encoder_degraded = copy.deepcopy(self.ae_model.encoder)
            self.quant_conv_degraded = copy.deepcopy(self.ae_model.quant_conv)
            self.quantize_shared = self.ae_model.quantize
            self.prompt_gen_module = PromptGenModule(prompt_dim=128, num_prompts=5)
        elif use_stable_diffusion:
            temp_vae = AutoencoderKL.from_pretrained(
                self.sd_model_name, subfolder="vae", torch_dtype=torch.float32
            )
            base_encoder = temp_vae.encoder
            base_encoder.load_state_dict(self.autoencoder.encoder.state_dict())
            if self.enable_promptir:
                self.encoder_degraded = PromptAwareEncoder(base_encoder=base_encoder, prompt_dim=128, num_prompts=5)
            else:
                self.encoder_degraded = base_encoder
            del temp_vae
        else:
            base_encoder = type(self.autoencoder.encoder)(
                **{k: v for k, v in self.autoencoder.encoder.__dict__.items() if not k.startswith('_')}
            )
            base_encoder.load_state_dict(self.autoencoder.encoder.state_dict())
            if self.enable_promptir:
                self.encoder_degraded = PromptAwareEncoder(base_encoder=base_encoder, prompt_dim=128, num_prompts=5)
            else:
                self.encoder_degraded = base_encoder

        if use_compvis_vqvae:
            for p in self.encoder_degraded.parameters():
                p.requires_grad = True
            for p in self.quant_conv_degraded.parameters():
                p.requires_grad = True
            for p in self.prompt_gen_module.parameters():
                p.requires_grad = True
        else:
            for p in self.encoder_degraded.parameters():
                p.requires_grad = True

    # diffusion helpers removed

    @staticmethod
    def _to_unet_channels(z3: torch.Tensor) -> torch.Tensor:
        if z3.shape[1] == 4:
            return z3
        pad = (0, 0, 0, 0, 0, 1)
        return torch.nn.functional.pad(z3, pad)

    @staticmethod
    def _from_unet_channels(z4: torch.Tensor) -> torch.Tensor:
        if z4.shape[1] >= 3:
            return z4[:, :3]
        raise ValueError("UNet output has <3 channels")

    # diffusion helpers removed

    # diffusion helpers removed

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _psnr(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        # x01, y01 in [0,1]
        mse = torch.mean((x01 - y01) ** 2)
        eps = 1e-8
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
        return psnr

    def forward(self, degraded_img, clean_img):
        if self.use_compvis_vqvae:
            _h_D = self.encoder_degraded(degraded_img)
            _h_D = self.quant_conv_degraded(_h_D)
            z_D, _, _ = self.quantize_shared(_h_D)

            _h_G = self.autoencoder.model.encoder(clean_img)
            _h_G = self.autoencoder.model.quant_conv(_h_G)
            z_G, _, _ = self.autoencoder.model.quantize(_h_G)
        elif self.use_stable_diffusion:
            encoder_output_D = self.encoder_degraded(degraded_img)
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

            posterior_D = DiagonalGaussianDistribution(encoder_output_D)
            z_D = posterior_D.sample()
            z_G = self.autoencoder.encode(clean_img).latent_dist.sample()
        else:
            z_D = self.encoder_degraded(degraded_img)
            z_G = self.autoencoder.encoder(clean_img)

        return z_D, z_G

    def training_step(self, batch, batch_idx):
        degraded_imgs, clean_imgs = batch
        z_D, z_G = self(degraded_imgs, clean_imgs)

        similarity = F.cosine_similarity(z_D, z_G, dim=1).mean()
        l2_loss = F.mse_loss(z_D, z_G)
        
        # Decode to get reconstructed images for image-level loss
        if self.use_stable_diffusion:
            D_out = self.autoencoder.decode(z_D).sample
            G_out = self.autoencoder.decode(z_G).sample
        elif self.use_compvis_vqvae:
            D_out = self.autoencoder.decode(z_D)
            G_out = self.autoencoder.decode(z_G)
        else:
            D_out = self.autoencoder.decoder(z_D)
            G_out = self.autoencoder.decoder(z_G)
        
        # Calculate image-level losses
        D_l2_loss = F.mse_loss(D_out, G_out)
        img_loss = D_l2_loss

        # Combined loss
        loss = (-self.similarity_weight * similarity + 
                self.l2_weight * l2_loss + 
                self.img_loss_weight * img_loss)
        
        # Logging
        self.log('train/similarity', similarity, prog_bar=True)
        self.log('train/l2_loss', l2_loss, prog_bar=True)
        self.log('train/img_loss', img_loss, prog_bar=True)
        self.log('train/D_l2_loss', D_l2_loss)
        self.log('train/total_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Support dict of loaders from per-degradation val
        if isinstance(batch, dict):
            # Lightning won't pass dict here; kept for safety
            return None
        degraded_imgs, clean_imgs = batch
        if (self.trainer.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):
            try:
                self._save_input_debug(degraded_imgs, clean_imgs)
            except Exception:
                pass
        z_D, z_G = self(degraded_imgs, clean_imgs)
        l2_loss = F.mse_loss(z_D, z_G)
        similarity = F.cosine_similarity(z_D, z_G, dim=1).mean()
        self.log('val/similarity', similarity, prog_bar=True)
        self.log('val/l2_loss', l2_loss, prog_bar=True)

        # PSNR and SSIM without diffusion (direct decode)
        with torch.no_grad():
            if self.use_stable_diffusion:
                restored_direct_D = self.autoencoder.decode(z_D).sample
                restored_direct_G = self.autoencoder.decode(z_G).sample
            elif self.use_compvis_vqvae:
                restored_direct_D = self.autoencoder.decode(z_D)
                restored_direct_G = self.autoencoder.decode(z_G)
            else:
                restored_direct_D = self.autoencoder.decoder(z_D)
                restored_direct_G = self.autoencoder.decoder(z_G)
            psnr_direct = self._psnr(self._to_01(restored_direct_D), self._to_01(clean_imgs))
            self.log('val/psnr_direct', psnr_direct, prog_bar=True)
            ssim_direct = self._ssim(self._to_01(restored_direct_D), self._to_01(clean_imgs))
            self.log('val/ssim_direct', ssim_direct, prog_bar=True)
            if (self.trainer.current_epoch % self.image_log_interval == 0) and (batch_idx == 0):    
                if self.use_stable_diffusion:
                    # diffusion on both degraded and clean latents
                    if self.supir_sampling_enabled:
                        # Use SUPIR-style sampling for better restoration quality
                        restored_diffusion_D = self.sd_latent_supir_sampling(
                            z_D, prompt="", 
                            restoration_guidance=self.supir_restoration_guidance,
                            noise_level=self.supir_noise_level
                        )
                        restored_diffusion_G = self.sd_latent_supir_sampling(
                            z_G, prompt="", 
                            restoration_guidance=self.supir_restoration_guidance,
                            noise_level=self.supir_noise_level
                        )
                    else:
                        # Use standard img2img sampling
                        restored_diffusion_D = self.sd_latent_img2img(z_D, prompt="")
                        restored_diffusion_G = self.sd_latent_img2img(z_G, prompt="")
                    
                    if self.color_correction_on:
                        # Color correction is already applied in SUPIR sampling if enabled
                        restored_diffusion_D = self.color_correction(restored_diffusion_D, degraded_imgs)
                    psnr_diffusion = self._psnr(self._to_01(restored_diffusion_D), self._to_01(clean_imgs))
                    psnr_key = 'val/psnr_supir' if self.supir_sampling_enabled else 'val/psnr_diffusion'
                    self.log(psnr_key, psnr_diffusion, prog_bar=True)
                    ssim_diffusion = self._ssim(self._to_01(restored_diffusion_D), self._to_01(clean_imgs))
                    ssim_key = 'val/ssim_supir' if self.supir_sampling_enabled else 'val/ssim_diffusion'
                    self.log(ssim_key, ssim_diffusion, prog_bar=True)
                    
                    # Log which sampling method is being used
                    if self.supir_sampling_enabled:
                        self.log('val/supir_noise_level', self.supir_noise_level)
                        self.log('val/supir_restoration_guidance', self.supir_restoration_guidance)
                    # Score refinement (optional visualization)
                    # restored_score_D = self.sd_latent_score_refine(z_D, prompt="", t_frac=0.5, refine_steps=10)
                    # restored_score_G = self.sd_latent_score_refine(z_G, prompt="", t_frac=0.5, refine_steps=10)
                    self._generate_restoration_samples_both_paths(
                        degraded_imgs, clean_imgs,
                        restored_direct_D, restored_diffusion_D,
                        restored_direct_G, restored_diffusion_G,
                    )
                    # self._generate_restoration_samples_score(
                    #     degraded_imgs, clean_imgs,
                    #     restored_direct_D, restored_score_D,
                    #     restored_direct_G, restored_score_G,
                    # )

        return similarity

    def color_correction(self, restored_img, degraded_img):
        # restored_img and degraded_img : [B, 3, H, W]
        xr_mean = restored_img.mean(dim=(2, 3), keepdim=True)
        xr_std = restored_img.std(dim=(2, 3), keepdim=True)
        xd_mean = degraded_img.mean(dim=(2, 3), keepdim=True)
        xd_std = degraded_img.std(dim=(2, 3), keepdim=True)

        return (restored_img - xr_mean) * (xd_std / xr_std) + xd_mean

    def _save_input_debug(self, degraded_imgs, clean_imgs):
        from torchvision.utils import save_image
        n_samples = min(4, degraded_imgs.size(0))
        if self.use_stable_diffusion or self.use_compvis_vqvae:
            degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
            clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
        else:
            degraded_display = torch.clamp((degraded_imgs[:n_samples] / 0.1) + 0.5, 0, 1)
            clean_display = torch.clamp((clean_imgs[:n_samples] / 0.1) + 0.5, 0, 1)

        grid = []
        for i in range(n_samples):
            grid.extend([degraded_display[i], clean_display[i]])

        os.makedirs(f"validation_samples_1/epoch_{self.trainer.current_epoch}", exist_ok=True)
        save_image(
            grid,
            f"validation_samples_1/epoch_{self.trainer.current_epoch}/inputs_degraded_vs_clean.png",
            nrow=2,
            normalize=False,
            padding=2,
        )

    def _generate_restoration_samples_both_paths(self, degraded_imgs, clean_imgs,
                                                 restored_direct_D, restored_diffusion_D,
                                                 restored_direct_G, restored_diffusion_G):
        try:
            with torch.no_grad():
                n_samples = min(4, degraded_imgs.size(0))
                degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_direct_D_display = torch.clamp((restored_direct_D[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_diffusion_D_display = torch.clamp((restored_diffusion_D[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_direct_G_display = torch.clamp((restored_direct_G[:n_samples] + 1.0) / 2.0, 0, 1)
                restored_diffusion_G_display = torch.clamp((restored_diffusion_G[:n_samples] + 1.0) / 2.0, 0, 1)
                
                comparison_imgs = []
                for i in range(n_samples):
                    target_h, target_w = clean_display[i].shape[-2], clean_display[i].shape[-1]

                    def resize_to(img, h, w):
                        if img.shape[-2] == h and img.shape[-1] == w:
                            return img
                        return F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

                    di = resize_to(degraded_display[i], target_h, target_w)
                    ci = resize_to(clean_display[i], target_h, target_w)
                    d_dir = resize_to(restored_direct_D_display[i], target_h, target_w)
                    d_dif = resize_to(restored_diffusion_D_display[i], target_h, target_w)
                    g_dir = resize_to(restored_direct_G_display[i], target_h, target_w)
                    g_dif = resize_to(restored_diffusion_G_display[i], target_h, target_w)

                    # layout: degraded | clean | D_direct | D_diff | G_direct | G_diff
                    comparison_imgs.extend([di, ci, d_dir, d_dif, g_dir, g_dif])

                output_dir = f"validation_samples_1/epoch_{self.trainer.current_epoch}"
                os.makedirs(output_dir, exist_ok=True)
                filename = "restoration_comparison_SUPIR.png" if self.supir_sampling_enabled else "restoration_comparison_inc_diffusion.png"
                save_image(comparison_imgs, f"{output_dir}/{filename}", nrow=6, normalize=False, padding=2)

        except Exception as e:
            print(f"Error generating restoration samples with diffusion: {e}")
                


    def _generate_restoration_samples_score(self, degraded_imgs, clean_imgs,
                                            restored_direct_D, restored_score_D,
                                            restored_direct_G, restored_score_G):
        try:
            with torch.no_grad():
                n_samples = min(4, degraded_imgs.size(0))
                degraded_display = torch.clamp((degraded_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                clean_display = torch.clamp((clean_imgs[:n_samples] + 1.0) / 2.0, 0, 1)
                rdir_D = torch.clamp((restored_direct_D[:n_samples] + 1.0) / 2.0, 0, 1)
                rsc_D = torch.clamp((restored_score_D[:n_samples] + 1.0) / 2.0, 0, 1)
                rdir_G = torch.clamp((restored_direct_G[:n_samples] + 1.0) / 2.0, 0, 1)
                rsc_G = torch.clamp((restored_score_G[:n_samples] + 1.0) / 2.0, 0, 1)

                comparison_imgs = []
                for i in range(n_samples):
                    target_h, target_w = clean_display[i].shape[-2], clean_display[i].shape[-1]

                    def resize_to(img, h, w):
                        if img.shape[-2] == h and img.shape[-1] == w:
                            return img
                        return F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

                    di = resize_to(degraded_display[i], target_h, target_w)
                    ci = resize_to(clean_display[i], target_h, target_w)
                    d_dir = resize_to(rdir_D[i], target_h, target_w)
                    d_sc = resize_to(rsc_D[i], target_h, target_w)
                    g_dir = resize_to(rdir_G[i], target_h, target_w)
                    g_sc = resize_to(rsc_G[i], target_h, target_w)

                    # layout: degraded | clean | D_direct | D_score | G_direct | G_score
                    comparison_imgs.extend([di, ci, d_dir, d_sc, g_dir, g_sc])

                output_dir = f"validation_samples_1/epoch_{self.trainer.current_epoch}"
                os.makedirs(output_dir, exist_ok=True)
                save_image(comparison_imgs, f"{output_dir}/restoration_comparison_score_refine.png", nrow=6, normalize=False, padding=2)

        except Exception as e:
            print(f"Error generating restoration samples (score refine): {e}")

    @torch.no_grad()
    def sd_latent_img2img(self, z_unscaled, prompt=""):
        """DDIM img2img with proper scaling and resizing for SD UNet compatibility."""
        device = z_unscaled.device
        batch_size = z_unscaled.shape[0]
        
        # 1) Get scaling factor correctly
        scaling = getattr(self.autoencoder.config, "scaling_factor", 0.18215)
        z = z_unscaled * scaling
        
        # 2) Handle UNet expected sample_size (e.g., 64 for SD 512)
        orig_h, orig_w = z.shape[-2], z.shape[-1]
        target_size = getattr(self.unet.config, "sample_size", 64)
        need_resize = (orig_h != target_size or orig_w != target_size)
        need_resize = False
        
        if need_resize:
            z_work = F.interpolate(z, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            z_work = z
        
        # 3) Text embeddings (CFG)
        uncond_inputs = self.tokenizer(["" for _ in range(batch_size)], padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
        uncond = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
        
        cond = None
        if prompt != "":  # Fix: use != instead of 'is not'
            text_inputs = self.tokenizer([prompt for _ in range(batch_size)], padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            cond = self.text_encoder(text_inputs.input_ids.to(device))[0]
        
        # 4) DDIM setup (img2img: add noise then denoise)
        self.scheduler.set_timesteps(self.diffusion_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Fix: proper strength handling (1.0 = full denoise, 0.0 = no denoise)
        t_start_idx = int(len(timesteps) * self.init_strength)
        t_start = timesteps[t_start_idx] if t_start_idx < len(timesteps) else timesteps[0]

        noise = torch.randn_like(z_work)
        z_t = self.scheduler.add_noise(z_work, noise, t_start)
        
        # 5) Denoise loop
        for t in timesteps:
            if t > t_start:
                continue
                
            if cond is not None and self.guidance_scale > 1.0:
                # Classifier-free guidance
                z_in = torch.cat([z_t, z_t], dim=0)
                if hasattr(self.scheduler, "scale_model_input"):
                    z_in = self.scheduler.scale_model_input(z_in, t)
                cond_in = torch.cat([uncond, cond], dim=0)
                noise_pred = self.unet(z_in, t, encoder_hidden_states=cond_in).sample
                noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            else:
                # No guidance
                if hasattr(self.scheduler, "scale_model_input"):
                    z_t = self.scheduler.scale_model_input(z_t, t)
                noise_pred = self.unet(z_t, t, encoder_hidden_states=uncond).sample
                
            z_t = self.scheduler.step(noise_pred, t, z_t).prev_sample
        
        # 6) Resize back if needed
        if need_resize:
            z_out = F.interpolate(z_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        else:
            z_out = z_t
            
        z_unscaled_out = z_out / scaling
        x_out = self.autoencoder.decode(z_unscaled_out).sample
        return x_out

    @torch.no_grad()
    def sd_latent_supir_sampling(self, z_unscaled, prompt="", restoration_guidance=1.5, noise_level=None):
        """SUPIR-style sampling with EDM sampler and restoration-specific guidance.
        Based on the SUPIR paper: https://arxiv.org/pdf/2401.13627
        """
        assert self.use_stable_diffusion, "SUPIR sampling only supported for Stable Diffusion path"
        device = z_unscaled.device
        scaling = getattr(self.autoencoder.config, "scaling_factor", 0.18215)
        z0 = z_unscaled * scaling
        batch_size, _, orig_h, orig_w = z0.shape
        
        # Handle UNet expected sample_size
        target_size = getattr(self.unet.config, "sample_size", 64)
        need_resize = (orig_h != target_size or orig_w != target_size)
        if need_resize:
            z_work = F.interpolate(z0, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            z_work = z0
        
        # Text embeddings for restoration guidance
        with torch.no_grad():
            # Negative prompt for degraded images
            negative_prompts = ["blurry, noise, artifacts, low quality" for _ in range(batch_size)]
            negative_inputs = self.tokenizer(negative_prompts, padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt")
            negative_embed = self.text_encoder(negative_inputs.input_ids.to(device))[0]
            
            # Positive prompt for high-quality restoration
            if prompt == "":
                prompt = "high quality, sharp, detailed, restored image"
            positive_prompts = [prompt for _ in range(batch_size)]
            positive_inputs = self.tokenizer(positive_prompts, padding="max_length",
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt")
            positive_embed = self.text_encoder(positive_inputs.input_ids.to(device))[0]
        
        # SUPIR-style diffusion with EDM sampler
        self.scheduler.set_timesteps(self.diffusion_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Initialize with controlled noise injection (SUPIR technique)
        noise_level = noise_level if noise_level is not None else self.supir_noise_level
        noise = torch.randn_like(z_work) * noise_level
        z_t = z_work + noise  # Direct noise addition instead of scheduler noise
        
        # Get starting timestep based on noise level
        t_start_idx = int(len(timesteps) * (1.0 - noise_level))
        t_start_idx = max(0, min(t_start_idx, len(timesteps) - 1))
        
        # EDM-style sampling loop with restoration guidance
        for i, t in enumerate(timesteps[t_start_idx:]):
            # Stochastic noise injection (EDM/SUPIR technique)
            if self.supir_s_tmin <= t <= self.supir_s_tmax and self.supir_s_noise > 0:
                gamma = min(self.supir_s_churn / (len(timesteps) - t_start_idx), 2**0.5 - 1)
                if gamma > 0:
                    eps = torch.randn_like(z_t) * self.supir_s_noise
                    z_t = z_t + eps * (gamma * ((t / 1000) ** 2)).sqrt()
            
            # Classifier-free guidance with restoration-specific conditioning
            z_in = torch.cat([z_t, z_t, z_t], dim=0)
            if hasattr(self.scheduler, "scale_model_input"):
                    z_in = self.scheduler.scale_model_input(z_in, t)
            cond_in = torch.cat([negative_embed, positive_embed, positive_embed], dim=0)
            
            # Predict noise with three conditions
            noise_pred = self.unet(z_in, t, encoder_hidden_states=cond_in).sample
            noise_negative, noise_positive, noise_positive2 = noise_pred.chunk(3, dim=0)
            
            # SUPIR-style guidance combination
            noise_pred = (noise_negative + 
                         restoration_guidance * (noise_positive - noise_negative) +
                         0.5 * (noise_positive2 - noise_positive))  # Additional refinement
            
            # Denoise step
            z_t = self.scheduler.step(noise_pred, t, z_t).prev_sample
            
            # Optional: Progressive sharpening (SUPIR technique)
            if i % 5 == 0 and i > 0:
                # Add small high-frequency details
                high_freq = z_t - F.avg_pool2d(F.avg_pool2d(z_t, 3, 1, 1), 3, 1, 1)
                z_t = z_t + 0.1 * high_freq
        
        # Resize back if needed
        if need_resize:
            z_out = F.interpolate(z_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        else:
            z_out = z_t
        
        z_unscaled_out = z_out / scaling
        x_out = self.autoencoder.decode(z_unscaled_out).sample
        
        return x_out


    def _psnr(self, x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        return psnr(x01, y01, data_range=1.0)

    def _ssim(self, x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
        return ssim(x01, y01, data_range=1.0)

    def configure_optimizers(self):
        if self.use_compvis_vqvae:
            trainable_params = list(self.encoder_degraded.parameters()) + list(self.quant_conv_degraded.parameters())
        else:
            trainable_params = self.encoder_degraded.parameters()
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/similarity"}}


class AlignmentDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, num_workers=0, train_split=0.95, use_compvis_vqvae=False, use_stable_diffusion=False, crop_size=256):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.use_compvis_vqvae = use_compvis_vqvae
        self.use_stable_diffusion = use_stable_diffusion
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        

        if self.use_compvis_vqvae or self.use_stable_diffusion:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def setup(self, stage=None):
        temp_full = DegradedCleanPairDataset(data_root=self.data_root, paired_transform=None, image_transform=None)
        train_size = int(self.train_split * len(temp_full))
        val_size = len(temp_full) - train_size
        temp_train, temp_val = torch.utils.data.random_split(temp_full, [train_size, val_size])
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


