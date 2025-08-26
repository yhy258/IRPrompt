import os
import argparse
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDIMScheduler
from aoiir.models.mod_controlnet import ModControlNet
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.tensorboard import SummaryWriter

from aoiir.datasets.multi_degradation import MultiDegradationDataset
from aoiir.models.zero_sft import ZeroSFTModulator


class ControlNetEmbedding(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor
    def forward(self, cond_img):
        enc_x = self.vae.encode(cond_img).latent_dist
        z_x = enc_x.sample() * self.scaling_factor
        return z_x.detach() # in order to avoid backprop to the encoder

class Captioner:
    def __init__(self, device: str = "cuda", positive_quality: Optional[str] = None):
        self.device = device
        self.positive_quality = positive_quality or (
            "cinematic, high contrast, highly detailed, ultra HD, photo-realistic, hyper sharpness"
        )
        self.backend = None
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            self.backend = "blip"
        except Exception:
            self.processor = None
            self.model = None

    @torch.no_grad()
    def __call__(self, image_tensor: torch.Tensor) -> List[str]:
        if self.backend != "blip" or image_tensor is None:
            # Return the positive_quality for each image in the batch (or 1 if not batched)
            n = image_tensor.shape[0] if image_tensor is not None and image_tensor.dim() == 4 else 1
            return [self.positive_quality] * n
        from torchvision.transforms.functional import to_pil_image
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        imgs = ((image_tensor.clamp(-1, 1) + 1.0) / 2.0).cpu()
        pil_imgs = [to_pil_image(img) for img in imgs]
        captions = []
        for pil_img in pil_imgs:
            try:
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                out = self.model.generate(**inputs, max_new_tokens=32)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            except Exception:
                caption = ""
            captions.append((caption + ", " if caption else "") + self.positive_quality)
        return captions


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def preprocess_control_image(img_tensor: torch.Tensor, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # img_tensor: [B,3,h,w] in [-1,1] -> [0,1], resized to (H,W)
    cond = (img_tensor + 1.0) / 2.0
    if cond.shape[1] == 1:
        cond = cond.repeat(1, 3, 1, 1)
    cond = F.interpolate(cond, size=(height, width), mode="bilinear", align_corners=False)
    return cond.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--controlnet_model", type=str, required=True, help="e.g., lllyasviel/sd-controlnet-canny")
    parser.add_argument("--data_root", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--lr_control", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=4)
    parser.add_argument("--val_steps", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/checkpoints/sd21_control_zerosft")
    parser.add_argument("--val_dir", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/validation_sd21_control_zerosft")
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--lambda_E", type=float, default=0.1)
    parser.add_argument("--allow_e_grad_from_diffusion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_logs"))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.precision == "fp16" else (torch.bfloat16 if args.precision == "bf16" else torch.float32)

    # Data
    dataset = MultiDegradationDataset(dataset_root=args.data_root, patch_size=256,
                                      de_type=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'lowlight'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dataset = Subset(dataset, list(range(min(64, len(dataset)))))
    val_loader = DataLoader(val_dataset, batch_size=args.val_samples, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    # Official diffusers ControlNet (expects RGB cond image)
    # controlnet: ControlNetModel = ControlNetModel.from_pretrained(args.controlnet_model)
    controlnet: ModControlNet = ModControlNet.from_unet(unet)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")

    vae.to(device)
    unet.to(device)
    controlnet.to(device)
    text_encoder.to(device)

    # Freeze
    freeze_module(unet)
    freeze_module(text_encoder)
    freeze_module(vae.decoder)
    freeze_module(vae.post_quant_conv)

    # Trainables
    zerosft = ZeroSFTModulator(text_hidden_dim=text_encoder.config.hidden_size).to(device)
    params_control = list(controlnet.parameters()) + list(zerosft.parameters())
    optim_control = torch.optim.AdamW(params_control, lr=args.lr_control, weight_decay=0.01)
    params_enc = list(vae.encoder.parameters()) + list(vae.quant_conv.parameters())
    optim_enc = torch.optim.AdamW(params_enc, lr=args.lr_encoder, weight_decay=0.0)

    captioner = Captioner(device=device)
    scaling_factor = vae.config.scaling_factor

    step = 0
    unet.eval(); text_encoder.eval(); vae.decoder.eval()

    def controlnet_unet_step(z_t, t, cond_img, text_emb, pooled_text):
        down_res, mid_res = controlnet(
            sample=z_t,
            timestep=t,
            encoder_hidden_states=text_emb,
            controlnet_cond=cond_img,
            return_dict=False,
        )
        down_res = list(down_res)
        for i in range(len(down_res)):
            down_res[i] = zerosft([down_res[i]], pooled_text)[0]
        mid_res = zerosft([mid_res], pooled_text)[0]
        noise_pred = unet(
            z_t,
            t,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample
        return noise_pred

    def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
        mse = F.mse_loss(x, y).item()
        if mse == 0:
            return 99.0
        return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
        # Very lightweight SSIM approx: average local variance contrast
        # For proper SSIM, integrate a robust implementation later
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = x.var()
        sigma_y = y.var()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
        return float(ssim.item())

    def validate(cur_step: int):
        controlnet.eval(); zerosft.eval(); unet.eval(); text_encoder.eval(); vae.decoder.eval()
        total_diff = 0.0; total_enc = 0.0; n = 0
        total_psnr = 0.0; total_ssim = 0.0
        with torch.no_grad():
            for batch_idx, (degrad_patch, clean_patch) in enumerate(val_loader):
                degrad_patch = degrad_patch.to(device)
                clean_patch = clean_patch.to(device)
                enc_x = vae.encode(clean_patch).latent_dist
                z_x = enc_x.sample() * scaling_factor
                bsz = z_x.shape[0]
                t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(z_x)
                alphas = scheduler.alphas_cumprod.to(device)
                a_t = alphas[t].view(bsz, 1, 1, 1)
                z_t = a_t.sqrt() * z_x + (1 - a_t).sqrt() * noise

                # Control image from reconstructed degraded: x_y = D(E(y))
                z_y = vae.encode(degrad_patch).latent_dist.sample() * scaling_factor
                x_y = vae.decode(z_y / scaling_factor).sample
                cond_img = preprocess_control_image(x_y, height=256, width=256, device=device, dtype=z_t.dtype)
                prompts = captioner(x_y)
                text = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
                text_emb = text_encoder(text.input_ids)[0]
                pooled_text = text_emb.mean(dim=1)

                ### change the controlnt's embdeder
                controlnet.controlnet_cond_embedding = ControlNetEmbedding(vae)
                noise_pred = controlnet_unet_step(z_t, t, cond_img, text_emb, pooled_text)
                loss_diff = F.mse_loss(noise_pred, noise)

                # encoder recon loss
                x_y_dec = vae.decode(z_y / scaling_factor).sample
                x_x_dec = vae.decode(z_x / scaling_factor).sample
                loss_enc = F.mse_loss(x_y_dec, x_x_dec)

                total_diff += loss_diff.item(); total_enc += loss_enc.item(); n += 1

                if batch_idx == 0:
                    scheduler.set_timesteps(args.val_steps)
                    z_t_s = torch.randn_like(z_x)
                    empty = tokenizer([""] * bsz, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
                    empty_emb = text_encoder(empty.input_ids)[0]
                    for tt in scheduler.timesteps:
                        eps_c = controlnet_unet_step(z_t_s, tt, cond_img, text_emb, pooled_text)
                        eps_u = controlnet_unet_step(z_t_s, tt, cond_img, empty_emb, pooled_text)
                        eps = eps_u + 4.0 * (eps_c - eps_u)
                        z_t_s = scheduler.step(eps, tt, z_t_s).prev_sample
                    x_hat = vae.decode(z_t_s / scaling_factor).sample
                    import torchvision.utils as vutils
                    grid = vutils.make_grid(torch.cat([(degrad_patch[:args.val_samples] + 1) / 2,
                                                       (clean_patch[:args.val_samples] + 1) / 2,
                                                       (x_hat[:args.val_samples].clamp(-1, 1) + 1) / 2], dim=0),
                                            nrow=args.val_samples)
                    from torchvision.utils import save_image
                    save_image(grid, os.path.join(args.val_dir, f"step_{cur_step:06d}.png"))
                # compute PSNR/SSIM on reconstructions
                # use x_hat vs clean (limited to first batch synth)
                x_hat_vis = (x_hat.clamp(-1, 1) + 1) / 2
                clean_vis = (clean_patch[:args.val_samples] + 1) / 2
                total_psnr += psnr(x_hat_vis, clean_vis)
                total_ssim += ssim_simple(x_hat_vis, clean_vis)
                n += 1
        avg_diff = total_diff/max(1,n); avg_enc = total_enc/max(1,n)
        avg_psnr = total_psnr/max(1,n); avg_ssim = total_ssim/max(1,n)
        print(f"[VAL {cur_step}] diff {avg_diff:.4f} | enc {avg_enc:.4f} | psnr {avg_psnr:.3f} | ssim {avg_ssim:.3f}")
        writer.add_scalar("val/diff_mse", avg_diff, cur_step)
        writer.add_scalar("val/enc_mse", avg_enc, cur_step)
        writer.add_scalar("val/psnr", avg_psnr, cur_step)
        writer.add_scalar("val/ssim", avg_ssim, cur_step)

    while step < args.max_steps:
        for degrad_patch, clean_patch in loader:
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            # prompts with dropout
            with torch.no_grad():
                z_y_vis = vae.encode(degrad_patch).latent_dist.sample() * scaling_factor
                x_y = vae.decode(z_y_vis / scaling_factor).sample
            base_prompts = captioner(x_y)
            bsz = degrad_patch.shape[0]
            prompts: List[str] = base_prompts
            if args.p_uncond > 0:
                mask = torch.rand(bsz, device=device) < args.p_uncond
                for i in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                    prompts[i] = ""
            text = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_emb = text_encoder(text.input_ids)[0]
                pooled_text = text_emb.mean(dim=1)

            # latents
            enc_x = vae.encode(clean_patch).latent_dist
            z_x = enc_x.sample() * scaling_factor
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(z_x)
            alphas = scheduler.alphas_cumprod.to(device)
            a_t = alphas[t].view(bsz, 1, 1, 1)
            z_x_det = z_x if args.allow_e_grad_from_diffusion else z_x.detach()
            z_t = a_t.sqrt() * z_x_det + (1 - a_t).sqrt() * noise

            # control image from reconstructed degraded
            z_y = vae.encode(degrad_patch).latent_dist.sample() * scaling_factor
            cond_img = preprocess_control_image(x_y, height=256, width=256, device=device, dtype=z_t.dtype)

            # train
            controlnet.train(); zerosft.train()
            controlnet.controlnet_cond_embedding = ControlNetEmbedding(vae)
            noise_pred = controlnet_unet_step(z_t, t, cond_img, text_emb, pooled_text)

            # diffusion loss
            pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
            if pred_type == "v_prediction":
                v_target = a_t.sqrt() * noise - (1 - a_t).sqrt() * z_x_det
                loss_diff = F.mse_loss(noise_pred, v_target)
            else:
                loss_diff = F.mse_loss(noise_pred, noise)

            # encoder recon loss
            enc_y2 = vae.encode(degrad_patch).latent_dist
            z_y2 = enc_y2.sample() * scaling_factor
            x_y_dec = vae.decode(z_y2 / scaling_factor).sample
            x_x_dec = vae.decode(z_x / scaling_factor).sample
            loss_enc = F.mse_loss(x_y_dec, x_x_dec)

            loss = loss_diff + args.lambda_E * loss_enc
            optim_control.zero_grad(set_to_none=True)
            optim_enc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_control, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(params_enc, max_norm=1.0)
            optim_control.step(); optim_enc.step()

            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f} | diff {loss_diff.item():.4f} | enc {loss_enc.item():.4f}")
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/diff_mse", float(loss_diff.item()), step)
                writer.add_scalar("train/enc_mse", float(loss_enc.item()), step)
            if step % args.save_every == 0 and step > 0:
                torch.save({
                    "controlnet": controlnet.state_dict(),
                    "zerosft": zerosft.state_dict(),
                    "vae_encoder": vae.encoder.state_dict(),
                    "vae_quant": vae.quant_conv.state_dict(),
                    "optim_control": optim_control.state_dict(),
                    "optim_enc": optim_enc.state_dict(),
                    "step": step,
                }, os.path.join(args.out_dir, f"step_{step}.pt"))
            if step % args.val_every == 0 and step > 0:
                validate(step)
                # controlnet.train(); zerosft.train(); unet.train(); text_encoder.train(); vae.decoder.train()

            step += 1
            if step >= args.max_steps:
                break

    torch.save({
        "controlnet": controlnet.state_dict(),
        "zerosft": zerosft.state_dict(),
        "vae_encoder": vae.encoder.state_dict(),
        "vae_quant": vae.quant_conv.state_dict(),
        "step": step,
    }, os.path.join(args.out_dir, "last.pt"))


if __name__ == "__main__":
    main()



