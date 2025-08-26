import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from aoiir.datasets.multi_degradation import MultiDegradationDataset
from aoiir.models.zero_sft import ZeroSFTModulator
from aoiir.models.latent_copied_controlnet import LatentCopiedControlNet


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--data_root", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--lr_control", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_steps", type=int, default=25)
    parser.add_argument("--val_samples", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/checkpoints/sd21_latent_controlnet")
    parser.add_argument("--val_dir", type=str, default="/home/joon/ImageRestoration-AllInOne/Combined_Method/validation_sd21_latent_controlnet")
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--lambda_E", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    # Copy encoder weights for latent ControlNet (custom, latent-accepting)
    controlnet = LatentCopiedControlNet(unet).to(device)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    # Freeze base models
    freeze(unet)
    freeze(text_encoder)
    freeze(vae.decoder)
    freeze(vae.post_quant_conv)

    zerosft = ZeroSFTModulator(text_hidden_dim=text_encoder.config.hidden_size).to(device)
    optim_control = torch.optim.AdamW(list(controlnet.parameters()) + list(zerosft.parameters()), lr=args.lr_control, weight_decay=0.01)
    optim_encoder = torch.optim.AdamW(list(vae.encoder.parameters()) + list(vae.quant_conv.parameters()), lr=args.lr_encoder)

    scaling = vae.config.scaling_factor
    step = 0

    def mid_residual_from_latent(z_cond: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor, pooled: torch.Tensor):
        # Run copied UNet on latent to produce mid residual only (trimmed control)
        _down_res, mid_res = controlnet(sample=z_cond, timestep=t, encoder_hidden_states=text_emb)
        mid_res = zerosft([mid_res], pooled)[0]
        return mid_res

    def validate(cur_step: int):
        controlnet.eval(); zerosft.eval(); unet.eval(); text_encoder.eval(); vae.decoder.eval()
        import torchvision.utils as vutils
        tot_d, tot_e, n = 0.0, 0.0, 0
        with torch.no_grad():
            for bidx, (y, x) in enumerate(val_loader):
                y = y.to(device); x = x.to(device)
                enc_x = vae.encode(x).latent_dist
                z_x = enc_x.sample() * scaling
                bsz = z_x.shape[0]
                t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(z_x)
                alphas = scheduler.alphas_cumprod.to(device)
                a_t = alphas[t].view(bsz, 1, 1, 1)
                z_t = a_t.sqrt() * z_x + (1 - a_t).sqrt() * noise

                z_y = vae.encode(y).latent_dist.sample() * scaling
                text = tokenizer(["quality, detailed, photorealistic"] * bsz, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
                text_emb = text_encoder(text.input_ids)[0]
                pooled = text_emb.mean(dim=1)

                mid_res = mid_residual_from_latent(z_y, t, text_emb, pooled)
                eps = unet(z_t, t, encoder_hidden_states=text_emb, mid_block_additional_residual=mid_res).sample
                tot_d += F.mse_loss(eps, noise).item(); n += 1

                if bidx == 0:
                    scheduler.set_timesteps(args.val_steps)
                    z_t_s = torch.randn_like(z_x)
                    empty = tokenizer([""] * bsz, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
                    empty_emb = text_encoder(empty.input_ids)[0]
                    for tt in scheduler.timesteps:
                        mres = mid_residual_from_latent(z_y, tt, text_emb, pooled)
                        eps_c = unet(z_t_s, tt, encoder_hidden_states=text_emb, mid_block_additional_residual=mres).sample
                        eps_u = unet(z_t_s, tt, encoder_hidden_states=empty_emb, mid_block_additional_residual=mres).sample
                        eps_all = eps_u + 4.0 * (eps_c - eps_u)
                        z_t_s = scheduler.step(eps_all, tt, z_t_s).prev_sample
                    x_hat = vae.decode(z_t_s / scaling).sample
                    grid = vutils.make_grid(torch.cat([(y[:args.val_samples] + 1) / 2, (x[:args.val_samples] + 1) / 2, (x_hat[:args.val_samples].clamp(-1, 1) + 1) / 2], dim=0), nrow=args.val_samples)
                    from torchvision.utils import save_image
                    save_image(grid, os.path.join(args.val_dir, f"step_{cur_step:06d}.png"))

                z_y2 = vae.encode(y).latent_dist.sample() * scaling
                tot_e += F.mse_loss(vae.decode(z_y2 / scaling).sample, vae.decode(z_x / scaling).sample).item()
        print(f"[VAL {cur_step}] diff {tot_d/max(1,n):.4f} | enc {tot_e/max(1,n):.4f}")

    while step < args.max_steps:
        for y, x in loader:
            y = y.to(device); x = x.to(device)

            bsz = y.shape[0]
            # text dropout
            text = tokenizer(["quality, detailed, photorealistic"] * bsz, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            if args.p_uncond > 0:
                mask = torch.rand(bsz, device=device) < args.p_uncond
                if mask.any():
                    empt = tokenizer([""] * int(mask.sum().item()), padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
                    text.input_ids[mask] = empt.input_ids
            with torch.no_grad():
                text_emb = text_encoder(text.input_ids)[0]
                pooled = text_emb.mean(dim=1)

            enc_x = vae.encode(x).latent_dist
            z_x = enc_x.sample() * scaling
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(z_x)
            alphas = scheduler.alphas_cumprod.to(device)
            a_t = alphas[t].view(bsz, 1, 1, 1)
            z_t = a_t.sqrt() * z_x.detach() + (1 - a_t).sqrt() * noise

            z_y = vae.encode(y).latent_dist.sample() * scaling

            controlnet.train(); zerosft.train()
            mres = mid_residual_from_latent(z_y, t, text_emb, pooled)
            eps = unet(z_t, t, encoder_hidden_states=text_emb, mid_block_additional_residual=mres).sample

            # diffusion loss
            pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
            if pred_type == "v_prediction":
                v_target = a_t.sqrt() * noise - (1 - a_t).sqrt() * z_x.detach()
                loss_diff = F.mse_loss(eps, v_target)
            else:
                loss_diff = F.mse_loss(eps, noise)

            # encoder recon loss
            z_y2 = vae.encode(y).latent_dist.sample() * scaling
            loss_enc = F.mse_loss(vae.decode(z_y2 / scaling).sample, vae.decode(z_x / scaling).sample)
            loss = loss_diff + args.lambda_E * loss_enc

            optim_control.zero_grad(set_to_none=True)
            optim_encoder.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(controlnet.parameters()) + list(zerosft.parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(vae.encoder.parameters()) + list(vae.quant_conv.parameters()), 1.0)
            optim_control.step(); optim_encoder.step()

            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f} | diff {loss_diff.item():.4f} | enc {loss_enc.item():.4f}")
            if step % args.save_every == 0 and step > 0:
                torch.save({
                    "controlnet": controlnet.state_dict(),
                    "zerosft": zerosft.state_dict(),
                    "vae_encoder": vae.encoder.state_dict(),
                    "vae_quant": vae.quant_conv.state_dict(),
                    "optim_control": optim_control.state_dict(),
                    "optim_encoder": optim_encoder.state_dict(),
                    "step": step,
                }, os.path.join(args.out_dir, f"step_{step}.pt"))
            if step % args.val_every == 0 and step > 0:
                validate(step)

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


