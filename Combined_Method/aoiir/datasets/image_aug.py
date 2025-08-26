"""
Created on 2020/9/8

@author: Boyun Li
"""
import os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import init
from PIL import Image
from torchvision import transforms


class Degradation(object):
    def __init__(self, patch_size):
        super(Degradation, self).__init__()
        self.toTensor = transforms.ToTensor()
        self.crop_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1

class EdgeComputation(nn.Module):
    def __init__(self, test=False):
        super(EdgeComputation, self).__init__()
        self.test = test
    def forward(self, x):
        if self.test:
            x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

            # y = torch.Tensor(x.size()).cuda()
            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, :, 1:] += x_diffx
            y[:, :, :, :-1] += x_diffx
            y[:, :, 1:, :] += x_diffy
            y[:, :, :-1, :] += x_diffy
            y = torch.sum(y, 1, keepdim=True) / 3
            y /= 4
            return y
        else:
            x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])

            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, 1:] += x_diffx
            y[:, :, :-1] += x_diffx
            y[:, 1:, :] += x_diffy
            y[:, :-1, :] += x_diffy
            y = torch.sum(y, 0) / 3
            y /= 4
            return y.unsqueeze(0)


# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
    assert len(image_size) > 1
    assert patches.shape[-3] == patches.shape[-2]
    H = image_size[0]
    W = image_size[1]
    patch_size = patches.shape[-2] - overlap
    image = np.zeros(image_size)
    idx = 0
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            image[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = patches[idx,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 :]
            idx += 1
    return image

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


# def random_augmentation(*args):
#     out = []
#     if random.randint(0, 1) == 1:
#         flag_aug = random.randint(1, 7)
#         for data in args:
#             out.append(data_augmentation(data, flag_aug).copy())
#     else:
#         for data in args:
#             out.append(data)
#     return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)