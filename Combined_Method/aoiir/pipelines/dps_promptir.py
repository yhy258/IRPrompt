import os
from typing import Optional, Tuple

from sympy.polys.polyroots import z
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from aoiir.models.promptir import LatentPromptAdapter, PromptAwareEncoder
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from einops import rearrange

class DPSPromptIRPipeline(nn.Module):
    """
    Diffusion Posterior Sampling in Stable Diffusion latent space with a
    PromptIR-style latent adapter F.

    - All SD2.1 modules (VAE, UNet, text) are frozen.
    - Only the adapter F is trainable (outside this inference class).
    - During sampling, we follow DPS update:
        z_{i-1} = z'_{i-1} - zeta_i * grad_{z_i} || F(z_D) - z0_hat(z_i) ||^2
      where z0_hat is predicted clean latent from current state using DDIM formula.
    """

    def __init__(
        self,
        sd_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        adapter: Optional[LatentPromptAdapter] = None,
        lambda_latent: float = 1.0,
        lambda_pixel: float = 1.0,
        scheduler_type: str = "ddim",
        device: str = "cuda",
    ) -> None:
        super(DPSPromptIRPipeline, self).__init__()
        self.device = device
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet", torch_dtype=torch.float32).to(device).eval()
        # self.scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        if scheduler_type == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        elif scheduler_type == "dpm":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")
        print("Prediction type of scheduler is: ", self.scheduler.config.prediction_type)
        print("init_noise_sigma of scheduler is: ", self.scheduler.init_noise_sigma)
        # self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder", torch_dtype=torch.float32).to(device).eval()
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")

        self.scaling = getattr(self.vae.config, "scaling_factor")

        self.pre = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.post = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.0) / 2.0),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        ])

        self.adapter = adapter if adapter is not None else LatentPromptAdapter()
        self.adapter.to(device)
        self.lambda_latent = lambda_latent
        self.lambda_pixel = lambda_pixel

        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def several_forwards(self, x_D, num_forwards: int = 5):
        x_D_repeat = x_D.unsqueeze(0).repeat(num_forwards, 1, 1, 1, 1)
        x_D_repeat = rearrange(x_D_repeat, 'n b c h w -> (n b) c h w')
        z_D = self.adapter(self.vae.encode(x_D_repeat).latent_dist.sample())
        z_D = rearrange(z_D, '(n b) c h w -> n b c h w', n=num_forwards)
        z_D = z_D * self.scaling
        return z_D



    # @torch.no_grad()
    # def encode(self, img: torch.Tensor) -> torch.Tensor:
    #     posterior = self.vae.encode(img).latent_dist
    #     z = posterior.sample() * self.scaling
    #     return z

    # @torch.no_grad()
    # def encode_stats(self, img: torch.Tensor):
    #     """
    #     Returns scaled mean and std of VAE posterior in latent space.
    #     """
    #     posterior = self.vae.encode(img).latent_dist
    #     # DiagonalGaussianDistribution exposes .mean and .logvar
    #     mu = posterior.mean * self.scaling
    #     std = torch.exp(0.5 * posterior.logvar) * self.scaling
    #     return mu, std

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.vae.decode(z / self.scaling, return_dict=False)[0]
        return x

    def compute_loss(self, x_D, x_G):
        # z_G = self.vae(x_G)
        z_D = self.adapter(self.vae.encode(x_D).latent_dist.sample())
        # z_D = self.adapter(self.vae.encode(x_D))
        # posterior_D = DiagonalGaussianDistribution(z_D)
        # z_D = posterior_D.sample()
        z_G = self.vae.encode(x_G).latent_dist.sample().detach()
        latent_loss = F.mse_loss(z_D, z_G)
        x_pred = self.vae.decode(z_D / self.scaling).sample
        x_gt = self.vae.decode(z_G / self.scaling).sample
        pixel_loss = F.mse_loss(x_pred, x_gt)
        total_loss = self.lambda_latent * latent_loss + self.lambda_pixel * pixel_loss
        ### in order to use the same scaling factor as the original code, we need to scale the z_D and z_G back to the original scale
        return z_D*self.scaling, z_G*self.scaling, total_loss,latent_loss, pixel_loss

    def forward(self, x_D, x_G):
        z_D, z_G, total_loss, latent_loss, pixel_loss = self.compute_loss(x_D, x_G)
        return z_D, z_G, total_loss, latent_loss, pixel_loss


    def _pred_noise(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # diffusion UNet expects scaled input for some schedulers
        if hasattr(self.scheduler, "scale_model_input"):
            z_in = self.scheduler.scale_model_input(z_t, t)
        else:
            z_in = z_t
        eps = self.unet(z_in, t, encoder_hidden_states=cond, return_dict=False)[0]
        return eps

    def _z0_from_eps(self, z_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Convert model output to x0 depending on prediction type
        device = z_t.device
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        abar = alphas_cumprod[t.long()].view(-1, 1, 1, 1)
        sqrt_abar = torch.sqrt(abar)
        sqrt_one_minus_abar = torch.sqrt(1.0 - abar)

        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if pred_type == "v_prediction":
            # x0 = alpha * z_t - sigma * v
            z0 = sqrt_abar * z_t - sqrt_one_minus_abar * eps
        else:
            # Default to epsilon prediction
            # x0 = (z_t - sqrt(1-abar)*eps) / sqrt(abar)
            z0 = (z_t - sqrt_one_minus_abar * eps) / sqrt_abar
        return z0

    def _score_from_eps(self, eps: torch.Tensor, t: torch.Tensor):
        device = eps.device
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        abar = alphas_cumprod[t.long()].view(-1, 1, 1, 1)
        # sqrt_abar = torch.sqrt(abar)
        sqrt_one_minus_abar = torch.sqrt(1.0 - abar)
        score = -1/sqrt_one_minus_abar * eps
        return score

    def _forward_diffuse_iterative(self, z0: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Iteratively forward-diffuse z0 to the first (largest) timestep in `timesteps`
        using q(z_tk | z_tk-1) with alpha_bar ratio between adjacent timesteps.

        This mirrors applying T decay steps instead of a one-shot add_noise.
        """
        with torch.no_grad():
            device = z0.device
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            # timesteps is typically descending for reverse. We need ascending for forward.
            ts = torch.flip(timesteps.to(device), dims=[0])
            z = z0.detach().clone()
            prev_abar = torch.tensor(1.0, device=device)
            for idx in range(ts.shape[0]):
                t = ts[idx]
                abar_t = alphas_cumprod[t.long()]
                ratio = (abar_t / prev_abar).view(1, 1, 1, 1)
                noise = torch.randn_like(z)
                z = torch.sqrt(ratio) * z + torch.sqrt(1.0 - ratio) * noise
                prev_abar = abar_t
            return z

    def _add_noise_to_latent(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add forward diffusion noise to z0 according to scheduler at timestep t.
        If scheduler exposes add_noise use it; otherwise fall back to alphas_cumprod.
        """
        if noise is None:
            noise = torch.randn_like(z0)
        if hasattr(self.scheduler, 'add_noise'):
            return self.scheduler.add_noise(z0, noise, t)
        device = z0.device
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        abar = alphas_cumprod[t.long()].view(-1, 1, 1, 1)
        return torch.sqrt(abar) * z0 + torch.sqrt(1.0 - abar) * noise

    def _init_latent_from_z0(self, z0: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Initialize z_t by forward-diffusing z0 to the first scheduler timestep.
        """
        bsz = z0.shape[0]
        t0 = timesteps[0]
        if not torch.is_tensor(t0):
            t0 = torch.tensor(t0, device=z0.device, dtype=torch.long)
        t = t0.repeat(bsz)
        return self._add_noise_to_latent(z0, t)



    @torch.no_grad()
    def _get_uncond_emb(self, batch: int) -> torch.Tensor:
        empty = self.tokenizer(["" for _ in range(batch)], padding="max_length",
                               max_length=self.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(empty.input_ids)[0]

    def get_positive_emb(self, batch: int) -> torch.Tensor:
        pos = np.random.choice(["masterpiece", "best quality", "high resolution", "realistic"])
        pos = 'best quality, high resolution, realistic'
        token = self.tokenizer([pos for _ in range(batch)], padding="max_length",
                               max_length=self.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(token.input_ids)[0]

    def get_negative_emb(self, batch: int) -> torch.Tensor:
        neg = np.random.choice(["oil painting", "cartoon", "blur", "dirty", "messy", "low quality", "deformation", "low resolution", "oversmooth"])
        neg = 'lowres, low quality, worst quality'

        token = self.tokenizer([neg for _ in range(batch)], padding="max_length",
                               max_length=self.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(token.input_ids)[0]

    def beyond_first_tweedie(
        self,
        z_D: torch.Tensor,
        num_steps: int = 50,
        stoc_avg_steps: int= 3,
        rand_num: int = 3,
        step_scale: float = 1.0,
        pixel_dps_scale: float = 1.0,
        second_order_scale: float = 1.0,
    ) -> torch.Tensor:
        bsz = z_D.shape[1]
        device = z_D.device
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Initialize z_t by iteratively forward-diffusing z_D to the first timestep
        z_t = self._forward_diffuse_iterative(z_D[0], timesteps) # z_D: N, B, C, H, W

        # Text conditioning: default to unconditional. Avoid randomized prompts.
        uncond = self._get_uncond_emb(bsz)


        for i, t in enumerate(timesteps):
            if t == 1000:
                continue
            for j, k in enumerate(range(stoc_avg_steps)):
                rand_eps = torch.randn_like(z_t.unsqueeze(0).repeat(rand_num, 1, 1, 1, 1)) # N, B, C, H, W

                with torch.enable_grad():
                    z_t_req = z_t.detach().clone().requires_grad_(True)

                    # Base prediction at current state (track grad w.r.t z_t_req)
                    eps_base = self._pred_noise(z_t_req, t, uncond)

                    # Pixel-space guidance (stop grad through UNet for this term)
                    z0_hat = self._z0_from_eps(z_t_req, eps_base.detach(), t)
                    x_pred = self.vae.decode(z0_hat / self.scaling, return_dict=False)[0]
                    x_gt = self.vae.decode(z_D / self.scaling, return_dict=False)[0]
                    pixel_loss = torch.linalg.norm(x_pred - x_gt)
                    grad_pixel = torch.autograd.grad(pixel_loss, z_t_req, retain_graph=True)[0]

                    # Second order correction: use perturbed latents derived from z_t_req
                    perturbed_req = z_t_req.unsqueeze(0) + rand_eps  # N, B, C, H, W
                    perturbed_req = rearrange(perturbed_req, 'n b c h w -> (n b) c h w')
                    uncond_perturbed = uncond.unsqueeze(0).repeat(rand_num, 1, 1, 1)
                    uncond_perturbed = rearrange(uncond_perturbed, 'n b l c -> (n b) l c')
                    eps_perturbed = self._pred_noise(perturbed_req, t, uncond_perturbed)
                    eps_perturbed = rearrange(eps_perturbed, '(n b) c h w -> n b c h w', n=rand_num)

                    inner_term = self._score_from_eps(eps_perturbed, t) - self._score_from_eps(eps_base.unsqueeze(0), t)
                    inner_term = rearrange(inner_term, 'n b c h w -> n b (c h w)', n=rand_num)
                    dim = inner_term.shape[-1]
                    flatten_perturbation = rearrange(rand_eps, 'n b c h w -> n b (c h w)')
                    inner_term_dot = torch.einsum('n b d, n b d -> n b', inner_term, flatten_perturbation)
                    
                    inner_term_dot = torch.mean(inner_term_dot)
                    second_correct = torch.autograd.grad(inner_term_dot, z_t_req, retain_graph=False)[0]

                    # Combine updates
                    z_t = z_t - pixel_dps_scale * grad_pixel - second_order_scale/dim * second_correct

            with torch.no_grad():
                eps = self._pred_noise(z_t, t, uncond)
                z_t = self.scheduler.step(eps, t, z_t, return_dict=False)[0]

        return z_t


    def pixel_space_dps_sample(
        self,
        z_D: torch.Tensor,
        num_steps: int = 50,
        step_scale: float = 1.0,
        dps_scale: float = 1.0,
        cfg: float = 0.,
        do_classifier_free_guidance: bool = False,
        manifold_eq: bool = False,
    ) -> torch.Tensor:
        """
        Run DPS in latent space. Returns restored latent z_0.
        """
        bsz = z_D.shape[1]
        device = z_D.device
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Initialize z_t by iteratively forward-diffusing z_D to the first timestep
        # z_t = self._forward_diffuse_iterative(z_D, timesteps)
        z_t = torch.randn(*z_D.shape[1:], device=device)

        # Text conditioning: default to unconditional. Avoid randomized prompts.
        if do_classifier_free_guidance:
            pos_uncond = self.get_positive_emb(bsz)
            neg_uncond = self.get_negative_emb(bsz)
            uncond = torch.cat([neg_uncond, pos_uncond], dim=0)
        else:
            uncond = self._get_uncond_emb(bsz)

        for i, t in enumerate(timesteps):
            if t == 1000:
                continue
            z_t_model = torch.cat([z_t] * 2) if do_classifier_free_guidance else z_t

            # Predict noise
            with torch.no_grad():
                eps = self._pred_noise(z_t_model, t, uncond)
            if do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = eps.chunk(2)
                # eps = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                neg, pos = eps.chunk(2)
                eps = neg + cfg * (pos - neg)
            # DDIM one step
            z_prime = self.scheduler.step(eps, t, z_t, return_dict=False)[0]
            # z_prime = step.prev_sample

            # Build differentiable path for measurement gradient (re-enable grads in val loop)
            if dps_scale > 0:
                with torch.enable_grad():
                    z_t_req = z_t.detach().clone().requires_grad_(True)
                    eps_const = eps.detach()
                    z0_hat = self._z0_from_eps(z_t_req, eps_const, t)
                    grad_comp = z0_hat if manifold_eq else z_t_req
                    difference = z_D - z0_hat.unsqueeze(0)
                    # DPS loss: ||F(z_D) - x0_hat||^2
                    norm = torch.linalg.norm(difference)
                    # norm = torch.mean((difference ** 2).sum(dim=(1,2,3,4)).sqrt())
                    grad = torch.autograd.grad(norm, grad_comp, retain_graph=False)[0]
                z_t = z_prime - dps_scale * grad
            
            else:
                # Pure diffusion sampling (no DPS): follow scheduler step
                z_t = z_prime
        return z_t


    def dps_sample(
        self,
        z_D: torch.Tensor, # N, B, C, H, W
        num_steps: int = 50,
        step_scale: float = 1.0,
        dps_scale: float = 1.0,
        cfg: float = 0.,
        do_classifier_free_guidance: bool = False,
        manifold_eq: bool = False,
    ) -> torch.Tensor:
        """
        Run DPS in latent space. Returns restored latent z_0.
        """
        n = z_D.shape[0]
        bsz = z_D.shape[1]
        device = z_D.device
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        z_t = torch.randn(*z_D.shape[1:], device=device) # B, C, H, W
        z_t = z_t * self.scheduler.init_noise_sigma

        # Text conditioning: default to unconditional. Avoid randomized prompts.
        if do_classifier_free_guidance:
            pos_uncond = self.get_positive_emb(bsz)
            neg_uncond = self.get_negative_emb(bsz)
            uncond = torch.cat([neg_uncond, pos_uncond], dim=0)
        else:
            uncond = self._get_uncond_emb(bsz)

        for i, t in enumerate(timesteps):
            if t == 1000:
                continue
            z_t_model = torch.cat([z_t] * 2) if do_classifier_free_guidance else z_t

            # Predict noise
            with torch.no_grad():
                eps = self._pred_noise(z_t_model, t, uncond)
            if do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = eps.chunk(2)
                # eps = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                neg, pos = eps.chunk(2)
                eps = neg + cfg * (pos - neg)
            # DDIM one step
            z_prime = self.scheduler.step(eps, t, z_t, return_dict=False)[0]
            # z_prime = step.prev_sample

            # Build differentiable path for measurement gradient (re-enable grads in val loop)
            if dps_scale > 0:
                with torch.enable_grad():
                    z_t_req = z_t.detach().clone().requires_grad_(True)
                    eps_const = eps.detach()
                    z0_hat = self._z0_from_eps(z_t_req, eps_const, t)
                    difference = z_D - z0_hat.unsqueeze(0)
                    # DPS loss: ||F(z_D) - x0_hat||^2
                    norm = torch.linalg.norm(difference)
                    # norm = torch.mean((difference ** 2).sum(dim=(1,2,3,4)).sqrt())
                    grad_comp = z0_hat if manifold_eq else z_t_req
                    grad = torch.autograd.grad(norm, grad_comp, retain_graph=False)[0]
                z_t = z_prime - dps_scale * grad
            
            else:
                # Pure diffusion sampling (no DPS): follow scheduler step
                z_t = z_prime
        return z_t


    def combined_dps_sample(
        self,
        z_D: torch.Tensor,
        num_steps: int = 50,
        latent_dps_scale: float = 1.0,
        pixel_dps_scale: float = 0.0,
        cfg: float = 0.,
        do_classifier_free_guidance: bool = False,
        manifold_eq: bool = False,
    ) -> torch.Tensor:
        """
        Combined DPS sampling that applies both latent-space and pixel-space guidance
        with independently tunable coefficients.

        - latent_dps_scale: coefficient for latent DPS grad on ||z_D - z0_hat||
        - pixel_dps_scale: coefficient for pixel DPS grad on ||x(z0_hat) - x(z_D)||
        """
        n = z_D.shape[0]
        bsz = z_D.shape[1]
        device = z_D.device
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Initialize z_t by iteratively forward-diffusing z_D to the first timestep
        # z_t = self._forward_diffuse_iterative(z_D, timesteps)
        z_t = torch.randn(*z_D.shape[1:], device=device)

        # Text conditioning
        if do_classifier_free_guidance:
            pos_uncond = self.get_positive_emb(bsz)
            neg_uncond = self.get_negative_emb(bsz)
            cond_embed = torch.cat([neg_uncond, pos_uncond], dim=0)
        else:
            cond_embed = self._get_uncond_emb(bsz)

        for i, t in enumerate(timesteps):
            if t == 1000:
                continue
            z_t_model = torch.cat([z_t] * 2) if do_classifier_free_guidance else z_t

            # Predict noise
            with torch.no_grad():
                eps = self._pred_noise(z_t_model, t, cond_embed)
            if do_classifier_free_guidance:
                neg, pos = eps.chunk(2)
                eps = neg + cfg * (pos - neg)

            # Diffusion step
            z_prime = self.scheduler.step(eps, t, z_t, return_dict=False)[0]

            # Guidance
            if latent_dps_scale > 0 or pixel_dps_scale > 0:
                with torch.enable_grad():
                    z_t_req = z_t.detach().clone().requires_grad_(True)
                    eps_const = eps.detach()
                    z0_hat = self._z0_from_eps(z_t_req, eps_const, t)

                    grad_comp = z0_hat if manifold_eq else z_t_req

                    total_grad = 0.0

                    # Latent DPS grad
                    if latent_dps_scale > 0:
                        latent_residual = z_D - z0_hat.unsqueeze(0)
                        latent_loss = torch.linalg.norm(latent_residual)
                        # latent_loss = torch.mean((latent_residual ** 2).sum(dim=(1,2,3,4)).sqrt())
                        retain = pixel_dps_scale > 0
                        grad_latent = torch.autograd.grad(latent_loss, grad_comp, retain_graph=retain)[0]
                        total_grad = total_grad + latent_dps_scale * grad_latent

                    # Pixel DPS grad
                    if pixel_dps_scale > 0:
                        x_pred = self.vae.decode(z0_hat / self.scaling, return_dict=False)[0]
                        x_gt = self.vae.decode(rearrange(z_D, 'n b c h w -> (n b) c h w') / self.scaling, return_dict=False)[0]
                        difference = x_pred.unsqueeze(0) - rearrange(x_gt, '(n b) c h w -> n b c h w', n=n)
                        pixel_loss = torch.linalg.norm(difference)
                        # pixel_loss = torch.mean((difference ** 2).sum(dim=(1,2,3,4)).sqrt())
                        grad_pixel = torch.autograd.grad(pixel_loss, grad_comp, retain_graph=False)[0]
                        total_grad = total_grad + pixel_dps_scale * grad_pixel

                z_t = z_prime - total_grad
            else:
                z_t = z_prime

        return z_t

    @torch.no_grad()
    def restore_path(self, image_path: str, num_steps: int = 50, out_path: Optional[str] = None) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        x = self.pre(img).unsqueeze(0).to(self.device)
        z_D = self.encode(x)
        z_rec = self.dps_sample(z_D, x, num_steps=num_steps)
        x_rec = self.decode(z_rec)
        x_rec = self.post(x_rec.squeeze(0))
        pil = transforms.ToPILImage()(x_rec)
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pil.save(out_path)
        return pil


