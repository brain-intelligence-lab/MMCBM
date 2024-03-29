# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os.path
import torch
import cv2

from params import standardization_int
from PIL import Image
import numpy as np
from monai.transforms import (
    ToTensor,
    RandRotate,
    Resize,
    RandZoom,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandShiftIntensity,
    RandCoarseDropout,
    ScaleIntensity,
    RandHistogramShift,
    Rand2DElastic,
    Rand3DElastic
)
from albumentations.augmentations import RandomBrightnessContrast


class BaseObject:
    def __init__(self):
        self.randomize = False
        self.R = np.random.RandomState()

    def do(self, prob):
        if self.R is None:
            return False
        return self.R.rand() < prob

    def operator(self, m, x):
        pass

    def operate(self, m, x):
        if isinstance(x, list):
            return [self.operator(m, p) for p in x]
        out = self.operator(m, x)
        self.randomize = False
        return out

    def __call__(self, data):
        imgs = dict(data)
        for m in imgs.keys():
            self.randomize = True
            imgs[m] = self.operate(m, imgs[m])
        return imgs


class LoadImage(BaseObject):
    def __init__(self, root_dir=None):
        super(LoadImage, self).__init__()
        self.root_dir = root_dir

    def operator(self, m, x):
        if self.root_dir is not None:
            x = os.path.join(self.root_dir, x)
        if isinstance(x, Image.Image):
            x = np.array(x)
        elif isinstance(x, str):
            x = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
        else:
            print(x)
            raise Exception
        return x


class ClipTrans(BaseObject):
    def __init__(self, trans=None):
        super(ClipTrans, self).__init__()
        self.trans = trans

    def operator(self, m, x):
        x = Image.open(x)
        return self.trans(x)


class ChannelFirstd(BaseObject):
    def __init__(self):
        super().__init__()

    def operator(self, m, x):
        return x.transpose(2, 0, 1)


class Resized(BaseObject):
    def __init__(self, img_size, mode='area'):
        super(Resized, self).__init__()
        self.resize = Resize(spatial_size=img_size, mode=mode)

    def operator(self, m, x):
        return self.resize(x)


class RandFlipd(BaseObject):
    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.flip = RandFlip(prob=prob, spatial_axis=1)
        self.flip_3d = RandFlip(prob=prob, spatial_axis=(1, 2))
        self.operator = lambda m, x: self.flip_3d(x, randomize=self.randomize) \
            if len(x.shape) == 4 else self.flip(x, randomize=self.randomize)


class RandZoomd(BaseObject):
    def __init__(self, min_zoom, max_zoom, prob):
        super().__init__()
        self.zoom = RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=prob, padding_mode='constant')
        self.operator = lambda m, x: self.zoom(x)


class RandGaussianNoised(BaseObject):
    def __init__(self, prob):
        super().__init__()
        self.noise = RandGaussianNoise(prob=prob)
        self.operator = lambda m, x: self.noise(x)


class RandGaussianSmoothd(BaseObject):
    def __init__(self, prob):
        super().__init__()
        self.smooth = RandGaussianSmooth(prob=prob)
        self.operator = lambda m, x: self.smooth(x)


class RandRotated(BaseObject):
    def __init__(self, range: float = .1, prob=.1):
        super().__init__()
        self.rotate = RandRotate(range_x=range, prob=prob, padding_mode='zeros')
        self.rotate_3d = RandRotate(range_x=range, range_y=range, range_z=range,
                                    prob=prob, padding_mode='zeros')
        self.operator = lambda m, x: self.rotate(x) if len(x.shape) == 3 else self.rotate_3d(x)


class ScaleIntensityd(BaseObject):
    def __init__(self, minv=0.0, maxv=1.0):
        super().__init__()
        self.scale = ScaleIntensity(minv=minv, maxv=maxv)
        self.operator = lambda m, x: self.scale(x)


class RandShiftIntensityd(BaseObject):
    def __init__(self, offsets=0.1, prob=1.0):
        super().__init__()
        self.shift = RandShiftIntensity(offsets=offsets, prob=prob)
        self.operator = lambda m, x: self.shift(x)


class RandCoarseDropoutd(BaseObject):
    def __init__(self, holes, spatial_size, prob=1.0):
        super().__init__()
        self.dropout = RandCoarseDropout(holes=holes, spatial_size=spatial_size, prob=prob)
        self.operator = lambda m, x: self.dropout(x)


class RandHistogramShiftd(BaseObject):
    def __init__(self, num_control_points=(5, 10), prob=.1):
        super().__init__()
        self.histo_shift = RandHistogramShift(num_control_points=num_control_points, prob=prob)
        self.operator = lambda m, x: self.histo_shift(x)


class RandElasticd(BaseObject):
    def __init__(self, spacing=(20, 30), magnitude_range=(0, 1), prob=.1):
        super().__init__()
        self.elastic = Rand2DElastic(spacing=spacing, magnitude_range=magnitude_range, padding_mode='zeros', prob=prob)
        self.elastic_3d = Rand3DElastic(sigma_range=(5, 8), magnitude_range=(20, 50), padding_mode='zeros',
                                        prob=prob)
        self.operator = lambda m, x: self.elastic(x) if len(x.shape) == 3 else self.elastic_3d(x)


class RandomBrightnessContrastd(BaseObject):
    def __init__(self, prob=.5):
        super().__init__()
        self.bs = RandomBrightnessContrast(p=prob)

    def operator(self, m, x):
        if len(x.shape) == 4:
            return x
        x = x.permute(1, 2, 0).numpy().astype(np.uint8)
        x = self.bs(image=x)['image']
        return x.transpose(2, 0, 1)


class ToTensord(BaseObject):
    def __init__(self, normalize=False, low=-1, high=1):
        super().__init__()
        self.normalize = normalize
        self.low = low
        self.high = high
        self.tensor = ToTensor(dtype=torch.float)

    def operator(self, m, x):
        x = self.tensor(x)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        if not isinstance(self.normalize, bool):
            x = self.normalize(x)
        elif isinstance(self.normalize, bool) and self.normalize:
            mean = torch.tensor(standardization_int[m]['mean']).reshape(3, 1, 1)
            std = torch.tensor(standardization_int[m]['std']).reshape(3, 1, 1)
            x = (x - mean) / std
        else:
            x = x * (self.high - self.low) + self.low
        return x
