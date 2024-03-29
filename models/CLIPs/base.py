# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch


class BaseCLIP:
    def _encode_image(self, image):
        raise NotImplementedError

    def encode_image(self, modality=None, image=None, keep_dict=False):
        if isinstance(image, dict):
            if modality != 'MM' and modality in image.keys():
                return self.encode_image(modality=modality, image=image[modality])
            if keep_dict:
                return {k: self.encode_image(modality=k, image=v) for k, v in image.items()}
            result = []
            for k, v in image.items():
                result.append(self.encode_image(modality=modality, image=v))
            return torch.max_pool1d(torch.concat(result, dim=1).permute(0, 2, 1), kernel_size=7).squeeze(-1)
        if image.ndim == 5:
            B = image.shape[0]
            image = image.reshape(-1, *image.shape[2:])
            embedding = self._encode_image(image)
            embedding = embedding.reshape(B, -1, embedding.shape[-1])
        else:
            embedding = self._encode_image(image)
        return {modality: embedding} if keep_dict else embedding

    def __getitem__(self, item):
        return {'encoder': self}

    def __call__(self, x):
        return self._encode_image(image=x)
