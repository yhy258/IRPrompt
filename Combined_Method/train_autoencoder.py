
import os
import sys
import argparse
import torchvision
from omegaconf import OmegaConf
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torchvision import transforms
from PIL import Image

# Add project directory to path for our custom module imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from taming.models.vqgan import Autoencoder # Our new model
from aoiir.datasets.clean import AllInOneCleanDataset # Our custom dataset

class ImageLogger(Callback):
    def __init__(self, epoch_frequency=10, max_images=4, clamp=True):
        super().__init__()
        self.epoch_freq = epoch_frequency
        self.max_images = max_images
        self.clamp = clamp

    def log_local(self, save_dir, images, current_epoch):
        root = os.path.join(save_dir, "images")
        os.makedirs(root, exist_ok=True)
        
        grid = torchvision.utils.make_grid(images, nrow=self.max_images)
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.detach().cpu().numpy()
        grid = (grid * 255).astype("uint8")
        filename = f"epoch_{current_epoch:04}.png"
        path = os.path.join(root, filename)
        Image.fromarray(grid).save(path)
        print(f"Saved validation grid to {path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.epoch_freq == 0:
            val_dataloader = trainer.datamodule.val_dataloader()
            val_batch = next(iter(val_dataloader))
            val_batch = val_batch.to(pl_module.device)

            pl_module.eval()
            with torch.no_grad():
                reconstructions, _ = pl_module(val_batch)
            pl_module.train()

            # Stack inputs and reconstructions
            # Take up to max_images
            inputs = val_batch[:self.max_images]
            recons = reconstructions[:self.max_images]
            
            if self.clamp:
                inputs = torch.clamp(inputs, -1.0, 1.0)
                recons = torch.clamp(recons, -1.0, 1.0)

            # Create a grid: original images on top, reconstructions on bottom
            grid = torch.cat((inputs, recons), 0)
            
            self.log_local(
                trainer.logger.save_dir,
                grid,
                trainer.current_epoch
            )


def instantiate_from_config(config):
    if not "target" in config:
        if config == {}:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    return getattr(__import__(module, fromlist=[cls]), cls)(**config.get("params", {}))

class DataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, num_workers=4, train_split=0.95):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        
        self.transform = transforms.Compose([
            transforms.RandomCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2. * x - 1.) # To be in range [-1, 1]
        ])

    def setup(self, stage=None):
        full_dataset = AllInOneCleanDataset(data_root=self.data_root, transform=self.transform)
        train_size = int(self.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Path to the root of the datasets.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    opt = parser.parse_args()

    config = OmegaConf.load(opt.config)

    # data
    data = DataModule(
        data_root=opt.data_root,
        batch_size=config.data.params.batch_size,
        num_workers=config.data.params.num_workers
    )

    # model
    model = Autoencoder(**config.model.params)

    # callbacks
    callbacks = []
    for cb_name, cb_conf in config.lightning.get("callbacks", {}).items():
        callbacks.append(instantiate_from_config(cb_conf))
    callbacks.append(ImageLogger(epoch_frequency=10, max_images=4)) # Add our image logger

    # trainer
    trainer_kwargs = {
        "callbacks": callbacks,
        "logger": instantiate_from_config(config.lightning.logger),
        **config.lightning.trainer
    }

    trainer = Trainer(**trainer_kwargs)

    # learning rate
    model.learning_rate = config.model.base_learning_rate
    print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run training
    print("Starting Autoencoder Training...")
    trainer.fit(model, datamodule=data, ckpt_path=opt.resume_from_checkpoint)
