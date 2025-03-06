import torch.utils.data as data
from torch.utils.data import DataLoader

import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from torchvision.transforms import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class SurgVisDom(data.Dataset):
    def __init__(self, 
                 list_file, 
                 labels_file,
                 num_segments=1, 
                 new_length=1,
                 image_tmpl='img_{:05d}.jpg', 
                 transform=None,
                 random_shift=True, 
                 test_mode=False, 
                 index_bias=0, 
                 kfold=False):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.kfold = kfold

        if self.index_bias is None:
             if self.image_tmpl == "frame{:d}.jpg":
                 self.index_bias = 0
             else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):
        if isinstance(self.image_tmpl, list):
            # 尝试多个图片模板格式
            for tmpl in self.image_tmpl:
                try:
                    image_path = os.path.join(directory, tmpl.format(idx))
                    if os.path.exists(image_path):
                        return [Image.open(image_path).convert('RGB')]
                except:
                    continue
            # 如果所有模板都失败，抛出错误
            raise ValueError(f"无法找到图片: {directory}, idx: {idx}, 尝试的模板: {self.image_tmpl}")
        else:
            # 原有的单一模板逻辑
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        # path, num_frames, label
        if not self.kfold:
            self.video_list = [VideoRecord(x.strip().split(' '))
                               for x in open(self.list_file)]
        else:
            self.video_list = [VideoRecord(x)
                               for x in self.list_file]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                # 循环采样时，确保起始索引不为0
                start_offset = randint(1, record.num_frames)  # 随机选择一个非零起始点
                return np.mod(np.arange(self.total_length) + start_offset, record.num_frames) + self.index_bias
            else:
                # 不循环采样时，确保索引不为0
                offsets = np.concatenate((
                    np.arange(1, record.num_frames),  # 从1开始
                    randint(1, record.num_frames, size=self.total_length - record.num_frames + 1)  # 随机补充索引
                ))
                return np.sort(offsets)[:self.total_length] + self.index_bias  # 截取前total_length个索引并排序
        else:
            # 视频帧数大于需要采样的总长度
            offsets = list()
            ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    # 确保采样起始点不为0
                    if tick == 0:
                        tick += randint(1, tick_len - self.seg_length + 1)
                    else:
                        tick += randint(0, tick_len - self.seg_length)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            if p == 0:
                p = 1  # 如果索引为0，则强制改为1
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


if __name__ == "__main__":
    train_list = '/home/heyuxin/anaconda3/envs/pytorch/SDA-CLIP-main/lists/surgvisdom/train_frames.txt'
    label_list = '/home/heyuxin/anaconda3/envs/pytorch/SDA-CLIP-main/lists/SurgVisDom_labels.csv'
    num_segments = 8
    image_tmpl = 'img_{:05d}.png'
    random_shift = True
    transform_train = None
    batch_size = 16
    workers = 0
    train_data = SurgVisDom(train_list, label_list,  # Todo train_list
                            num_segments = num_segments, image_tmpl = image_tmpl,
                            random_shift = random_shift, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              num_workers=workers, shuffle=True, pin_memory=False, drop_last=True)
    for x in train_loader:
        print(x)