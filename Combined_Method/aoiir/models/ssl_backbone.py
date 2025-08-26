import torch
import torch.nn as nn


def _normalize_model_name(name: str) -> str:
    # Map common aliases/suffixes to timm's canonical names
    aliases = {
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "vit_base_patch14_reg4_dinov2.lvd142m": "vit_base_patch14_dinov2",
        "vit_b_dinov2": "vit_base_patch14_dinov2",
        "dinov2-base": "vit_base_patch14_dinov2",
    }
    if name in aliases:
        return aliases[name]
    # Strip known suffixes if present
    for suf in [".lvd142m", ".reg4", ".reg4_dinov2", "_reg4_dinov2", ".dinov2", "_dinov2"]:
        if name.endswith(suf):
            name = name.replace(suf, "")
    # Ensure includes dinov2 token
    if "dinov2" in name and "patch14" in name and "vit_base" in name:
        return "vit_base_patch14_dinov2"
    return name


class SSLBackbone(nn.Module):
    """Thin wrapper around a self-supervised ViT (e.g., DINOv2 via timm).

    Outputs a dict with:
      - 'global': (B, D) global embedding (CLS or pooled)
      - 'tokens': (B, S, D) token sequence if available, else None

    Notes
    - Input expected in [-1,1] or [0,1]; will be normalized to ImageNet stats.
    - Model parameters are frozen by default.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backend = None
        self.timm_model = None
        self.hf_model = None
        self.output_dim = None

        norm_name = _normalize_model_name(model_name)

        # Try timm first
        try:
            import timm  # type: ignore
            self.timm_model = timm.create_model(
                norm_name,
                pretrained=pretrained,
                num_classes=0,
                features_only=False,
            )
            self.backend = "timm"
            self.output_dim = getattr(self.timm_model, "num_features", None)
        except Exception:
            # Try HF Dinov2
            try:
                from transformers import AutoImageProcessor, Dinov2Model  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Neither timm nor transformers Dinov2 is available. Install `timm` or `transformers`."
                ) from exc

            # Map common vit_base_patch14_dinov2 -> facebook/dinov2-base
            hf_name = {
                "vit_base_patch14_dinov2": "facebook/dinov2-base",
                "dinov2-base": "facebook/dinov2-base",
            }.get(norm_name, model_name if model_name.startswith("facebook/") else "facebook/dinov2-base")

            self.image_processor = AutoImageProcessor.from_pretrained(hf_name)
            self.hf_model = Dinov2Model.from_pretrained(hf_name)
            self.backend = "hf"
            self.output_dim = int(self.hf_model.config.hidden_size)

        # Register normalization buffers (default ImageNet; HF overrides below)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if self.backend == "hf":
            try:
                mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
                std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
            except Exception:
                pass
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

        if freeze:
            if self.backend == "timm":
                for p in self.timm_model.parameters():
                    p.requires_grad = False
                self.timm_model.eval()
            else:
                for p in self.hf_model.parameters():
                    p.requires_grad = False
                self.hf_model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        # Accept [-1,1] or [0,1]
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        x = (x - self.imagenet_mean) / self.imagenet_std

        # timm ViT forward_features may return Tensor or dict
        if self.backend == "timm":
            feats = self.timm_model.forward_features(x)

            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    global_vec = feats["x_norm_clstoken"]
                elif "cls_token" in feats:
                    global_vec = feats["cls_token"]
                else:
                    maybe_tokens = feats.get("x_norm_patchtokens", None)
                    if maybe_tokens is not None and maybe_tokens.ndim == 3:
                        global_vec = maybe_tokens.mean(dim=1)
                    else:
                        t = next((v for v in feats.values() if isinstance(v, torch.Tensor)), None)
                        if t is None:
                            raise RuntimeError("Unexpected feats structure from SSL backbone (timm)")
                        global_vec = t.mean(dim=tuple(range(1, t.ndim)))
                token_seq = feats.get("x_norm_patchtokens", None)
            else:
                if feats.ndim == 2:
                    global_vec = feats
                    token_seq = None
                elif feats.ndim == 3:
                    global_vec = feats[:, 0]
                    token_seq = feats
                else:
                    global_vec = feats.mean(dim=(2, 3))
                    token_seq = None
        else:
            # HF Dinov2 backend
            outputs = self.hf_model(pixel_values=x)
            hidden = outputs.last_hidden_state  # (B, S, D)
            global_vec = hidden[:, 0]
            token_seq = hidden

        return {"global": global_vec, "tokens": token_seq}


