import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from tqdm import tqdm

def augment_hsv(im, hgain=0.1, sgain=0.5, vgain=0.5, p=0.5):
    # HSV color-space augmentation
    rdn = np.random.rand()
    im_augment = im
    if (hgain or sgain or vgain) and (rdn < p):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        im_augment = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return im_augment

def add_noise(img, mean=0, std=150, p=0.5):
    # add white noise
    rdn = np.random.rand()
    img_noise = img
    if rdn < p:
        noise = np.random.normal(mean, std, img.shape).astype(np.int8)
        img_noise = img + noise
        img_noise = np.clip(img_noise, 0, 255)
    return img_noise

def augment_flip_0(img, p=0.5):
    rdn = np.random.rand()
    img_flip = img
    if rdn < p:
        img_flip = np.flip(img, axis=0)
    return img_flip

def augment_flip_1(img, p=0.5):
    rdn = np.random.rand()
    img_flip = img
    if rdn < p:
        img_flip = np.flip(img, axis=1)
    return img_flip

def augment_rotate(img, max_angle=10, p=0.5):
    rdn = np.random.rand()
    img_rotated = img
    if rdn < p:
        angle = np.random.uniform(low=-max_angle, high=max_angle)
        img_rotated = ndimage.rotate(img, angle, reshape=False, mode='mirror')
    return img_rotated


class data_augmentation:
    def __init__(self, image_train_folder: str,
                       hgain: float = 0.1,
                       sgain: float = 0.5,
                       vgain: float = 0.5,
                       noise_mean: float = 0,
                       noise_std: float = 150,
                       max_angle_rotate: float = 10):
        
        self.list_img_files = [os.path.join(image_train_folder, file)
                                for file in os.listdir(image_train_folder) 
                                if file.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        self.num_image = len(self.list_img_files)

        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.max_angle_rotate = max_angle_rotate

    def run(self, num_images_gen: int,
                  save_folder: str, 
                  p_hsv: float = 0.5,
                  p_noise: float = 0.5,
                  p_flip0: float = 0.5,
                  p_flip1: float = 0.5,
                  p_rotate: float = 0.5):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for i in tqdm(range(num_images_gen)):
            idx_img = np.random.randint(0, self.num_image)
            img = cv2.imread(self.list_img_files[idx_img])
            img = augment_flip_0(img, p=p_flip0)
            img = augment_flip_1(img, p=p_flip1)
            img = augment_hsv(img, hgain=self.hgain, sgain=self.sgain, vgain=self.vgain, p=p_hsv)
            img = add_noise(img, mean=self.noise_mean, std=self.noise_std, p=p_noise)
            img = augment_rotate(img, max_angle=self.max_angle_rotate, p=p_rotate)
            save_file = 'aug_' + str(i) + '_.jpg'
            cv2.imwrite(os.path.join(save_folder, save_file), img)