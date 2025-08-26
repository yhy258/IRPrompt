import os
import glob
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from aoiir.datasets.image_aug import random_augmentation, Degradation, crop_img


class PairedRandomCropFlip:
    """
    Apply the same random resize-to-min-side, crop, and flips to a pair of PIL images.
    Ensures both outputs have exactly crop_size.
    """

    def __init__(self, crop_size=(256, 256)):
        self.crop_size = crop_size

    def _ensure_min_side(self, img, min_side):
        h, w = img.size[1], img.size[0]
        if min(h, w) < min_side:
            scale = float(min_side) / float(min(h, w))
            new_h = max(int(round(h * scale)), min_side)
            new_w = max(int(round(w * scale)), min_side)
            img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)
        return img

    def __call__(self, img_a, img_b):
        crop_h, crop_w = self.crop_size
        min_side = min(crop_h, crop_w)
        img_a = self._ensure_min_side(img_a, min_side)
        img_b = self._ensure_min_side(img_b, min_side)

        i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=self.crop_size)
        img_a = TF.crop(img_a, i, j, h, w)
        img_b = TF.crop(img_b, i, j, h, w)

        if random.random() < 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
        if random.random() < 0.5:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)
        return img_a, img_b


class PairedCenterCrop:
    """Deterministic paired center-crop with ensure-min-side."""

    def __init__(self, crop_size=(256, 256)):
        self.crop_size = crop_size

    def _ensure_min_side(self, img, min_side):
        h, w = img.size[1], img.size[0]
        if min(h, w) < min_side:
            scale = float(min_side) / float(min(h, w))
            new_h = max(int(round(h * scale)), min_side)
            new_w = max(int(round(w * scale)), min_side)
            img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)
        return img

    def __call__(self, img_a, img_b):
        crop_h, crop_w = self.crop_size
        min_side = min(crop_h, crop_w)
        img_a = self._ensure_min_side(img_a, min_side)
        img_b = self._ensure_min_side(img_b, min_side)
        img_a = TF.center_crop(img_a, self.crop_size)
        img_b = TF.center_crop(img_b, self.crop_size)
        return img_a, img_b


class DegradedCleanPairDataset(torch.utils.data.Dataset):
    """
    Provides pairs of (degraded, clean) images across multiple datasets.
    """

    def __init__(self, data_root, paired_transform=None, image_transform=None):
        super().__init__()
        self.data_root = data_root
        self.paired_transform = paired_transform
        self.image_transform = image_transform
        self.pairs = []

        self._collect_pairs()
        print(f"Total number of degraded-clean pairs: {len(self.pairs)}")

    def _collect_pairs(self):
        self._collect_denoising_pairs()
        self._collect_dehazing_pairs()
        self._collect_deraining_pairs()
        self._collect_lowlight_pairs()
        self._collect_deblurring_pairs()

    def _collect_denoising_pairs(self):
        bsd_clean = glob.glob(os.path.join(self.data_root, "BSD400", "*.jpg"))
        for clean_path in bsd_clean:
            self.pairs.append((clean_path, clean_path))

        wed_clean = glob.glob(os.path.join(self.data_root, "WED", "*.bmp"))
        for clean_path in wed_clean:
            self.pairs.append((clean_path, clean_path))

    def _collect_dehazing_pairs(self):
        indoor_degraded = glob.glob(os.path.join(self.data_root, "SOTS", "indoor", "hazy", "*.png"))
        for degraded_path in indoor_degraded:
            clean_path = degraded_path.replace("hazy", "gt")
            if os.path.exists(clean_path):
                self.pairs.append((degraded_path, clean_path))

        outdoor_degraded = glob.glob(os.path.join(self.data_root, "SOTS", "outdoor", "hazy", "*.png"))
        for degraded_path in outdoor_degraded:
            clean_path = degraded_path.replace("hazy", "gt")
            if os.path.exists(clean_path):
                self.pairs.append((degraded_path, clean_path))

    def _collect_deraining_pairs(self):
        rain_images = glob.glob(os.path.join(self.data_root, "Rain100L", "rain-*.png"))
        for rain_path in rain_images:
            norain_path = rain_path.replace("rain-", "norain-")
            if os.path.exists(norain_path):
                self.pairs.append((rain_path, norain_path))

    def _collect_lowlight_pairs(self):
        low_images = glob.glob(os.path.join(self.data_root, "lol_dataset", "our485", "low", "*.png"))
        for low_path in low_images:
            high_path = low_path.replace("low", "high")
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

    def _collect_deblurring_pairs(self):
        blur_images = glob.glob(os.path.join(self.data_root, "gopro", "train", "**", "blur", "*.png"), recursive=True)
        for blur_path in blur_images:
            sharp_path = blur_path.replace("blur", "sharp")
            if os.path.exists(sharp_path):
                self.pairs.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        degraded_path, clean_path = self.pairs[idx]
        try:
            degraded_img = Image.open(degraded_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")

            if self.paired_transform is not None:
                degraded_img, clean_img = self.paired_transform(degraded_img, clean_img)

            if self.image_transform:
                # Apply transforms with identical RNG seed to keep stochastic ops in sync (if any)
                seed = torch.seed()
                torch.manual_seed(seed)
                degraded_img = self.image_transform(degraded_img)
                torch.manual_seed(seed)
                clean_img = self.image_transform(clean_img)

            if isinstance(degraded_img, torch.Tensor):
                degraded_img = degraded_img.contiguous().clone()
            if isinstance(clean_img, torch.Tensor):
                clean_img = clean_img.contiguous().clone()

            return degraded_img, clean_img
        except Exception as e:
            print(f"Could not load pair: {degraded_path}, {clean_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self))






# 5D 'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5 lowlight
class TrainDataset5D(torch.utils.data.Dataset):
    def __init__(self, data_file_dir='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/data_dir', 
            dataset_root='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset',  
            patch_size=512, 
            de_type=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'lowlight']
        ):
        super(TrainDataset5D, self).__init__()
        self.data_file_dir = data_file_dir
        self.dataset_root = dataset_root
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(patch_size)
        self.de_temp = 0
        self.de_type = de_type
        self.patch_size = patch_size
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5 , 'lowlight' : 6}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
        ])

        self.toTensor = transforms.ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type: # ---------
            self._init_blur_ids()
        if 'lowlight' in self.de_type: # ---------
            self._init_lol_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        
        # ref_file = self.data_file_dir + "noisy/denoise_airnet.txt"
        # ref_file = self.data_file_dir + "noisy/denoise.txt"
        # temp_ids = []
        # temp_ids+= [id_.strip() for id_ in open(ref_file)]
        # clean_ids = []
        # name_list = os.listdir(self.args.denoise_dir)
        # clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]
        clean_ids = []
    
        # BSD400 폴더에서 이미지 가져오기
        bsd_dir = os.path.join(self.dataset_root, "BSD400")
        if os.path.exists(bsd_dir):
            bsd_files = os.listdir(bsd_dir)
            # 이미지 확장자만 필터링
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            bsd_images = [f for f in bsd_files if any(f.lower().endswith(ext) for ext in image_extensions)]
            clean_ids += [os.path.join(bsd_dir, img_name) for img_name in bsd_images]
            # print(f"Found {len(bsd_images)} images in BSD400")
        
        # WED 폴더에서 이미지 가져오기  
        wed_dir = os.path.join(self.dataset_root, "WED")
        if os.path.exists(wed_dir):
            wed_files = os.listdir(wed_dir)
            # 이미지 확장자만 필터링
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            wed_images = [f for f in wed_files if any(f.lower().endswith(ext) for ext in image_extensions)]
            clean_ids += [os.path.join(wed_dir, img_name) for img_name in wed_images]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.data_file_dir + "rainy/rainTrain_trim.txt"
        temp_ids+= [self.dataset_root + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 80

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.data_file_dir + "hazy/train_hazy_trim.txt"
        temp_ids+= [self.dataset_root + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_blur_ids(self): # ------
        temp_ids = []
        blur = self.data_file_dir + "gopro/train_gopro_trim.txt"
        temp_ids+= [self.dataset_root + id_.strip() for id_ in open(blur)]
        self.blur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.blur_ids = self.blur_ids *30

        self.blur_counter = 0
        
        self.num_blur = len(self.blur_ids)
        print("Total blur Ids : {}".format(self.num_blur))    

    def _init_lol_ids(self): # ------
        temp_ids = []
        lol = self.data_file_dir + "lol/train_lol.txt"
        temp_ids+= [self.dataset_root + id_.strip() for id_ in open(lol)]
        self.lol_ids = [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.lol_ids = self.lol_ids*60

        self.lol_counter = 0
        
        self.num_lol = len(self.lol_ids)
        print("Total lol Ids : {}".format(self.num_lol)) 

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def _get_rainy_gt_name(self, rainy_name):
        seg_list = rainy_name.split(os.path.sep)
        gt_name = os.path.join(seg_list[0], 'no'+seg_list[-1])
        # gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split(os.path.sep)[:2]
        file_name = hazy_name.split(os.path.sep)[-1]
        nonhazy_name = os.path.join(*dir_name, 'gt', file_name)
        
        # dir_name = hazy_name.split("synthetic")[0] + 'original/'
        # name = file_name.split('_')[0]
        # suffix = '.' + file_name.split('.')[-1]
        # nonhazy_name = dir_name + name + suffix
        return nonhazy_name
    
    def _get_sharp_name(self, blur_name): # ------get no blur
        path_seg = blur_name.split('blur')
        sharp_name = os.path.join(path_seg[0], 'sharp', path_seg[-1])
        # sharp_name = blur_name.split("blur")[0] + 'sharp/' + blur_name.split('/')[-1]
        return sharp_name
    
    def _get_light_name(self, lol_name): # ------ get no blur
        path_seg = lol_name.split('low')
        light_name = os.path.join(path_seg[0], 'high', path_seg[-1])
        # light_name = lol_name.split("low")[0] + 'high/' + lol_name.split('/')[-1]
        return light_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids

        if "deblur" in self.de_type: # -------
            self.sample_ids+= self.blur_ids

        if "lowlight" in self.de_type: # ------
            self.sample_ids+= self.lol_ids
        
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_rainy_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # blur Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_sharp_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 6:
                # lowlight enhancement
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_light_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)


            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)
