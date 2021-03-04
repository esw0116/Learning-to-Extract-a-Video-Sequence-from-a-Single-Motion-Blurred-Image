import glob
import imageio
import numpy as np
import os
import random
import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader

from data import preprocessing


class VideoBase(data.Dataset):
    def __init__(self, data_root, img_type, training):
        super(VideoBase, self).__init__()
        self.img_type = img_type
        self.training = training
        self.patch_size = 256

        self._set_directory(data_root)
        self.data_dict = self._scan(training)

        # Pre-decode png files
        if self.img_type == 'bin':
            for k in tqdm.tqdm(self.data_dict.keys(), ncols=80):
                bin_path = os.path.join(self.data_root, 'bin')
                for idx, v in enumerate(self.data_dict[k]):
                    save_as = v.replace(self.data_root, bin_path)
                    save_as = save_as.replace('.png', '')
                    # If we don't have the binary, make it.
                    if not os.path.isfile(save_as+'.npy'):
                        os.makedirs(os.path.dirname(save_as), exist_ok=True)
                        img = imageio.imread(v)
                        # Bypassing the zip archive error
                        # _, w, c = img.shape
                        # dummy = np.zeros((1,w,c))
                        # img_dummy = np.concatenate((img, dummy), axis=0)
                        # torch.save(img_dummy, save_as)
                        np.save(save_as, img)
                    # Update the dictionary
                    self.data_dict[k][idx] = save_as + '.npy'
        
        self.n_samples = 0
        self.n_sample_list = []
        # when testing, we do not overlap the video sequence (0~6, 7~13, ...)
        if training:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) - (self.seq_len - 1)
            self.n_sample_list.append(self.n_samples)
        else:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) // self.seq_len
            self.n_sample_list.append(self.n_samples)
            
        print(self.n_sample_list)

    def _find_key(self, idx):
        for i, k in enumerate(self.data_dict.keys()):
            if self.n_sample_list[i] <= idx and idx < self.n_sample_list[i+1]:
                return k, idx - self.n_sample_list[i]
        
        raise ValueError()

    def __getitem__(self, idx):
        key, index = self._find_key(idx)
        if self.training:
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
        else:
            index *= self.seq_len
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
        if self.training:
            r = random.random()
            if r > 0.5:
                filepath_list.reverse()

        if self.img_type == 'img':
            fn_read = imageio.imread
        elif self.img_type == 'bin':
            fn_read = np.load
        else:
            raise ValueError('Wrong img type: {}'.format(self.img_type))
        
        imgs = [fn_read(f) for f in filepath_list]
        #imgs = np.stack(imgs, axis=-1)
        
        if self.training:
            crop_imgs = preprocessing.np2tensor(*imgs)
            crop_imgs = preprocessing.common_crop(*crop_imgs, patch_size=self.patch_size)
            crop_imgs = preprocessing.augment(*crop_imgs)
            crop_imgs = torch.stack(crop_imgs, dim=0)

            return crop_imgs

        else:
            imgs = preprocessing.np2tensor(*imgs)
            imgs = torch.stack(imgs, dim=0)
            return imgs, filepath_list

    def __len__(self):
        return self.n_samples
