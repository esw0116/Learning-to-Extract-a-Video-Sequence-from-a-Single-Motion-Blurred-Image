import glob
import os
from os import path
import random

from data import redbase

import numpy as np
import imageio

import torch
import torch.utils.data as data


class REDS120fps(redbase.RedBase):
    """GOPRO train OR test subset class
    """
    def __init__(self, *args, **kwargs):
        super(REDS120fps, self).__init__(*args, **kwargs)

    def _set_directory(self, data_root):
        self.data_root = path.join(data_root, 'REDS120fps')

    def _scan(self, training):
        def _make_keys(dir_path):
            """
            :param dir_path: Path
            :return: train_000 form
            """
            dir, base = path.dirname(dir_path), path.basename(dir_path)
            tv = 'train' if dir.find('train')>=0 else 'test'
            return tv + '_' + base


        if training:
            dir_train = path.join(self.data_root, 'train/train_orig')
            list_seq = glob.glob(dir_train+'/*')
            data_dict = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + '.png'))
                ) for k in list_seq
            }

        else:
            dir_test = path.join(self.data_root, 'val/val_orig')
            list_seq = glob.glob(dir_test+'/*')
            data_dict = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + '.png'))
                ) for k in list_seq
            }

        return data_dict