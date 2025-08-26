import os
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionRestorationPipeline:
    def __init__(self, encoder_degraded_path, sd_model_name="stabilityai/stable-diffusion-2-1-base", device="cuda"):
        self.device = device
        self.sd_model_name = sd_model_name
        self._load_models(encoder_degraded_path)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2. * x - 1.),
        ])
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.) / 2.),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),
        ])

    def _load_models(self, encoder_degraded_path):
        self.vae = AutoencoderKL.from_pretrained(self.sd_model_name, subfolder="vae", torch_dtype=torch.float32)
        self.vae.to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(self.sd_model_name, subfolder="unet", torch_dtype=torch.float32)
        self.unet.to(self.device).eval()
        self.scheduler = DDIMScheduler.from_pretrained(self.sd_model_name, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(self.sd_model_name, subfolder="text_encoder", torch_dtype=torch.float32)
        self.text_encoder.to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sd_model_name, subfolder="tokenizer")

        if encoder_degraded_path != "none":
            checkpoint = torch.load(encoder_degraded_path, map_location=self.device)
            temp_vae = AutoencoderKL.from_pretrained(self.sd_model_name, subfolder="vae", torch_dtype=torch.float32)
            self.encoder_degraded = temp_vae.encoder
            self.encoder_degraded.load_state_dict(checkpoint['encoder_state_dict'])
            del temp_vae
        else:
            self.encoder_degraded = self.vae.encoder
        self.encoder_degraded.to(self.device).eval()

    def encode_degraded(self, degraded_img):
        with torch.no_grad():
            encoder_output = self.encoder_degraded(degraded_img)
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            posterior = DiagonalGaussianDistribution(encoder_output)
            z_D = posterior.sample() * self.vae.config.scaling_factor
        return z_D

    def apply_diffusion(self, z_D, num_steps=50, guidance_scale=7.5):
        with torch.no_grad():
            self.scheduler.set_timesteps(num_steps)
            noise = torch.randn_like(z_D)
            timesteps = self.scheduler.timesteps
            init_timestep = timesteps[int(len(timesteps) * 0.8)]
            z_t = self.scheduler.add_noise(z_D, noise, init_timestep)
            for t in timesteps:
                if t > init_timestep:
                    continue
                latent_model_input = z_t
                text_inputs = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
                z_t = self.scheduler.step(noise_pred, t, z_t, return_dict=False)[0]
            return z_t

    def decode_restored(self, z_restored):
        with torch.no_grad():
            z_restored = z_restored / self.vae.config.scaling_factor
            return self.vae.decode(z_restored).sample

    def restore_image(self, image_path, output_path=None, num_diffusion_steps=50, guidance_scale=7.5):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        z_D = self.encode_degraded(image_tensor)
        z_restored = self.apply_diffusion(z_D, num_diffusion_steps, guidance_scale)
        restored_tensor = self.decode_restored(z_restored)
        restored_tensor = self.reverse_transform(restored_tensor.squeeze(0))
        restored_image = transforms.ToPILImage()(restored_tensor)
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_restored{input_path.suffix}"
        restored_image.save(output_path)
        return restored_image





