import torch
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from aoiir.models.ssl_backbone import SSLBackbone
from aoiir.models.ssl_adapter import SSLAdapter


class SSLRestorationPipeline:
    def __init__(
        self,
        sd_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        ssl_model_name: str = "vit_base_patch14_dinov2",
        num_ssl_tokens: int = 8,
        clip_hidden: int = 1024,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae", torch_dtype=torch.float16).to(device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet", torch_dtype=torch.float16).to(device).eval()
        self.scheduler = DDIMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder", torch_dtype=torch.float16).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")

        self.ssl_backbone = SSLBackbone(model_name=ssl_model_name, pretrained=True, freeze=True).to(device)
        in_dim = getattr(self.ssl_backbone, "output_dim", 768)
        self.ssl_adapter = SSLAdapter(in_dim=in_dim, out_dim=clip_hidden, num_tokens=num_ssl_tokens).to(device).eval()

        self.scaling = getattr(self.vae.config, "scaling_factor", 0.18215)
        self.pre = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.post = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.0) / 2.0),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        ])

    @torch.no_grad()
    def infer(self, image_path: str, num_steps: int = 50, strength: float = 0.8, guidance_scale: float = 1.0, prompt: str = ""):
        img = Image.open(image_path).convert("RGB")
        x = self.pre(img).unsqueeze(0).to(self.device)

        # SSL context
        ssl_out = self.ssl_backbone(x)
        ssl_tokens = self.ssl_adapter(ssl_out["global"])  # (1, T, C)

        # text context
        uncond_inputs = self.tokenizer([""], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]
        if prompt:
            cond_inputs = self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            cond = self.text_encoder(cond_inputs.input_ids.to(self.device))[0]
        else:
            cond = None
        ctx_uncond = torch.cat([ssl_tokens, uncond], dim=1)
        ctx_cond = torch.cat([ssl_tokens, cond], dim=1) if cond is not None else None

        # encode → latent
        posterior = self.vae.encode(x).latent_dist
        z_unscaled = posterior.sample()
        z = z_unscaled * self.scaling

        # img2img schedule
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        t_start = timesteps[int(len(timesteps) * strength)]
        noise = torch.randn_like(z)
        z_t = self.scheduler.add_noise(z, noise, t_start)

        for t in timesteps:
            if t > t_start:
                continue
            if ctx_cond is not None and guidance_scale > 1.0:
                z_in = torch.cat([z_t, z_t], dim=0)
                if hasattr(self.scheduler, "scale_model_input"):
                    z_in = self.scheduler.scale_model_input(z_in, t)
                cond_in = torch.cat([ctx_uncond, ctx_cond], dim=0)
                eps = self.unet(z_in, t, encoder_hidden_states=cond_in).sample
                eps_u, eps_c = eps.chunk(2, dim=0)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                z_in = z_t
                if hasattr(self.scheduler, "scale_model_input"):
                    z_in = self.scheduler.scale_model_input(z_in, t)
                eps = self.unet(z_in, t, encoder_hidden_states=ctx_uncond).sample
            z_t = self.scheduler.step(eps, t, z_t).prev_sample

        x_rec = self.vae.decode(z_t / self.scaling).sample
        x_rec = self.post(x_rec.squeeze(0))
        return transforms.ToPILImage()(x_rec)


