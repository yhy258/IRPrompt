import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from aoiir.datasets.image_aug import random_augmentation, crop_img, Degradation
from aoiir.datasets.paired import PairedRandomCropFlip, PairedCenterCrop


class MultiDegradationDataset(torch.utils.data.Dataset):
    """
    Multi-degradation dataset for SSL training and encoder alignment.
    Supports denoise, derain, dehaze, deblur, lowlight enhancement.
    """
    
    def __init__(
        self,
        data_file_dir='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/data_dir',
        dataset_root='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset',
        patch_size=256,
        de_type=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'lowlight'],
    ):
        super().__init__()
        self.data_file_dir = data_file_dir
        self.dataset_root = dataset_root
        self.patch_size = patch_size
        self.de_type = de_type
        # self.paired_transform = paired_transform
        # self.image_transform = image_transform
        # self.use_synthetic_noise = use_synthetic_noise
        
        # Degradation simulator for synthetic noise
        # if use_synthetic_noise:
        self.D = Degradation(patch_size)
        
        self.de_dict = {
            'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2,
            'derain': 3, 'dehaze': 4, 'deblur': 5, 'lowlight': 6
        }
        
        self.sample_ids = []
        self._init_ids()
        self._merge_ids()

        self.crop_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
        ])

        self.toTensor = transforms.ToTensor()
        
        print(f"Total samples: {len(self.sample_ids)}")

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
        rs = os.path.join(self.data_file_dir,  "rainy/rainTrain_trim.txt")
        temp_ids+= [self.dataset_root+os.path.sep + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 80

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = os.path.join(self.data_file_dir, "hazy/train_hazy_trim.txt")
        temp_ids+= [self.dataset_root + os.path.sep + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_blur_ids(self): # ------
        temp_ids = []
        blur = os.path.join(self.data_file_dir, "gopro/train_gopro_trim.txt")
        temp_ids+= [self.dataset_root + os.path.sep + id_.strip() for id_ in open(blur)]
        self.blur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.blur_ids = self.blur_ids *30

        self.blur_counter = 0
        
        self.num_blur = len(self.blur_ids)
        print("Total blur Ids : {}".format(self.num_blur))    

    def _init_lol_ids(self): # ------
        temp_ids = []
        lol = os.path.join(self.data_file_dir, "lol/train_low_trim.txt")
        temp_ids+= [self.dataset_root + os.path.sep + id_.strip() for id_ in open(lol)]
        self.lol_ids = [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.lol_ids = self.lol_ids*60

        self.lol_counter = 0
        
        self.num_lol = len(self.lol_ids)
        print("Total lol Ids : {}".format(self.num_lol)) 

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        # 이미지가 patch_size보다 작으면 resize
        if H < self.patch_size or W < self.patch_size:
            scale = max(self.patch_size / H, self.patch_size / W)
            new_H, new_W = int(H * scale) + 1, int(W * scale) + 1
            img_1 = np.array(Image.fromarray(img_1).resize((new_W, new_H), Image.BICUBIC))
            img_2 = np.array(Image.fromarray(img_2).resize((new_W, new_H), Image.BICUBIC))
            H, W = new_H, new_W
        
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def _get_rainy_gt_name(self, rainy_name):
        seg_list = rainy_name.split('rainy')
        gt_name = seg_list[0] + 'no' + seg_list[-1][1:]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        seg_list = hazy_name.split('hazy')
        dir_name = seg_list[0] + 'gt'
        hazy_file_name = seg_list[-1][1:]
        name = hazy_file_name.split('_')[0]
        suffix = '.jpg'
        nonhazy_name = os.path.join(dir_name , name + suffix)
        return nonhazy_name
    
    def _get_sharp_name(self, blur_name): # ------get no blur
        path_seg = blur_name.split('blur')
        sharp_name = path_seg[0] + 'sharp' + path_seg[-1]
        # sharp_name = os.path.join(path_seg[0], 'sharp', path_seg[-1])
        # sharp_name = blur_name.split("blur")[0] + 'sharp/' + blur_name.split('/')[-1]
        return sharp_name
    
    def _get_light_name(self, lol_name): # ------ get no blur
        path_seg = lol_name.split('low')
        light_name = path_seg[0] + 'high' + path_seg[-1]
        # light_name = os.path.join(path_seg[0], 'high', path_seg[-1])
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
        
        if de_id < 3:  # Synthetic noise (denoise_15, denoise_25, denoise_50)
            clean_id = sample["clean_id"]
            
            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
                
        else:  # Real degradation pairs
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

        clean_patch = self.toTensor(clean_patch) * 2 -1 # [-1, 1]
        degrad_patch = self.toTensor(degrad_patch) * 2 -1 # [-1, 1]
        
        
        return degrad_patch, clean_patch


    def __len__(self):
        return len(self.sample_ids)


# Factory function for easy integration
def create_multi_degradation_dataset(
    data_root,
    patch_size=256,
    degradation_types=None,
):
    """
    Create MultiDegradationDataset with appropriate transforms for SSL/alignment training.
    
    Args:
        data_root: Root directory containing datasets
        patch_size: Patch size for cropping
        degradation_types: List of degradation types to include
    """
    if degradation_types is None:
        degradation_types = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'lowlight']
    
    return MultiDegradationDataset(
        dataset_root=data_root,
        patch_size=patch_size,
        de_type=degradation_types,
    )
