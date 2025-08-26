import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from aoiir.models.zero_sft import ZeroSFTModulator


def ensure_nchw_and_resize(cond: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if cond.dim() != 4:
        raise ValueError("control condition tensor must be 4D [B,C,H,W] or [B,H,W,C]")
    if cond.shape[-1] in (1, 3, 4) and cond.shape[1] not in (1, 3, 4):
        cond = cond.permute(0, 3, 1, 2).contiguous()
    if cond.shape[-2:] != ref.shape[-2:]:
        cond = torch.nn.functional.interpolate(cond, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return cond

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(device)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet").to(device)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder").to(device)
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    try:
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4).to(device)
        cond_in_latent = True
    except TypeError:
        controlnet = ControlNetModel.from_unet(unet).to(device)
        cond_in_latent = False

    zerosft = ZeroSFTModulator(text_hidden_dim=text_encoder.config.hidden_size).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    controlnet.load_state_dict(ckpt["controlnet"], strict=False)
    zerosft.load_state_dict(ckpt["zerosft"], strict=False)
    vae.encoder.load_state_dict(ckpt["vae_encoder"], strict=False)
    vae.quant_conv.load_state_dict(ckpt["vae_quant"], strict=False)

    scheduler = DDIMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")

    preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ])
    post = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1.0) / 2.0),
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
    ])

    img = Image.open(args.input).convert("RGB")
    y = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_y = vae.encode(y).latent_dist
        z_y = enc_y.sample() * vae.config.scaling_factor
        x_y = vae.decode(z_y / vae.config.scaling_factor).sample

    # Minimal caption: you can plug the training Captioner here if needed
    prompt = "cinematic, high contrast, highly detailed, ultra HD, photo-realistic, hyper sharpness"
    text = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    text_emb = text_encoder(text.input_ids)[0]
    pooled_text = text_emb.mean(dim=1)

    # ControlNet residuals on-the-fly will be computed per step using z_t and z_y
    scheduler.set_timesteps(args.steps)
    z_t = torch.randn_like(z_y)
    timesteps = scheduler.timesteps

    # Prepare unconditional embeddings
    empty = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
    empty_emb = text_encoder(empty.input_ids)[0]

    for t in timesteps:
        z_y_match = ensure_nchw_and_resize(z_y, z_t)
        down_res, mid_res = controlnet(
            sample=z_t,
            timestep=t,
            encoder_hidden_states=text_emb,
            controlnet_cond=z_y_match,
            return_dict=False,
        )
        down_res = list(down_res)
        for i in range(len(down_res)):
            down_res[i] = zerosft([down_res[i]], pooled_text)[0]
        mid_res = zerosft([mid_res], pooled_text)[0]

        noise_cond = unet(
            z_t,
            t,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample

        noise_uncond = unet(z_t, t, encoder_hidden_states=empty_emb).sample
        noise_pred = noise_uncond + args.cfg * (noise_cond - noise_uncond)
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample

    z_0 = z_t
    x = vae.decode(z_0 / vae.config.scaling_factor).sample
    x = post(x.squeeze(0))
    out = transforms.ToPILImage()(x.cpu())
    out.save(args.output)


if __name__ == "__main__":
    main()


