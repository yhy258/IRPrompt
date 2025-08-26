import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from taming.models.vqgan import VQModel

class CompVisVQVAE(pl.LightningModule):
    """
    CompVis VQ-f4 VQVAE wrapper.
    """

    def __init__(self, model_path="model.ckpt", image_key="image"):
        super().__init__()
        self.image_key = image_key
        self.model_path = model_path
        self._load_model()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self):
        try:
            self.model = VQModel.load_from_checkpoint(self.model_path, strict=True)
            self.model.eval()
        except Exception as e:
            print(f"Error loading CompVis VQ-f4 model with Lightning method: {e}")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            ddconfig = {
                'double_z': False,
                'z_channels': 3,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 4,
                'ch': 128,
                'ch_mult': [1, 2, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.0
            }
            lossconfig = {"target": "torch.nn.Identity"}
            self.model = VQModel(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=3, n_embed=8192)
            state_dict = checkpoint['state_dict']
            model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
            self.model.load_state_dict(model_state_dict, strict=False)
            self.model.eval()

    def encode(self, x):
        with torch.no_grad():
            quant, _, _ = self.model.encode(x)
            return quant

    def decode(self, z):
        with torch.no_grad():
            dec = self.model.decode(z)
            return dec

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, input):
        z = self.encode(input)
        dec = self.decode(z)
        return dec, z
