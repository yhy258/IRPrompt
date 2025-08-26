import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class AllInOneCleanDataset(Dataset):
    """
    Loads only clean (ground truth) images across multiple datasets
    for training the clean autoencoder.
    """

    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []

        self._collect_paths()
        self.image_paths = sorted(list(set(self.image_paths)))
        print(f"Total number of unique clean images: {len(self.image_paths)}")

    def _collect_paths(self):
        # 1. Denoising
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "BSD400", "*.jpg")))
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "WED", "*.bmp")))

        # 2. Dehazing (SOTS)
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "SOTS", "indoor", "gt", "*.png")))
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "SOTS", "outdoor", "gt", "*.png")))

        # 3. Deraining (Rain100L)
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "Rain100L", "norain-*.png")))

        # 4. Low-light (LOL-v1)
        self.image_paths.extend(glob.glob(os.path.join(self.data_root, "lol_dataset", "our485", "high", "*.png")))

        # 5. Deblurring (GoPro)
        self.image_paths.extend(
            glob.glob(
                os.path.join(self.data_root, "gopro", "train", "**", "sharp", "*.png"),
                recursive=True,
            )
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Could not load image: {img_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self))





