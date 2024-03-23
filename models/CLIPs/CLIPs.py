import os
import torch
import numpy as np

###################################################### MedKLip ##############################################
from PIL import Image
from transformers import CLIPFeatureExtractor, AutoTokenizer
from transformers.image_transforms import convert_to_rgb, to_channel_dimension_format
from utils.dataloader import clip_transforms
from .base import BaseCLIP


class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self,
                 do_resize=True,
                 size=224,
                 resample=Image.BICUBIC,
                 do_center_crop=True,
                 crop_size=224,
                 do_normalize=True,
                 do_convert_rgb=False,
                 **kwargs):
        from medclip import constants
        image_mean = constants.IMG_MEAN
        image_std = constants.IMG_STD
        super().__init__(do_resize=do_resize, size=size, resample=resample,
                         do_center_crop=do_center_crop, crop_size=crop_size,
                         do_normalize=do_normalize, image_mean=image_mean,
                         image_std=image_std,
                         do_convert_rgb=do_convert_rgb, **kwargs)

    def __call__(self,
                 image,
                 **kwargs):
        image = np.asarray(image)
        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            image = convert_to_rgb(image)
        # if self.do_pad_square:
        #     images = [self.pad_img(image, min_size=self.size) for image in images]

        if self.do_resize and self.size is not None and self.resample is not None:
            image = self.resize(image=image, size=self.size, resample=self.resample)

        if self.do_center_crop and self.crop_size is not None:
            image = self.center_crop(image, self.crop_size)

        if self.do_rescale:
            image = self.rescale(image=image, scale=self.rescale_factor)

        if self.do_normalize:
            image = self.normalize(image=image, mean=self.image_mean, std=self.image_std)
        image = to_channel_dimension_format(image, 'channels_first')
        return torch.from_numpy(image).float()


class Clip(BaseCLIP):
    def __init__(self, device, clip_name, download_root):
        self.device = device
        from clip import clip
        self.model, self._transforms = clip.load(name=clip_name.split('_')[1], device=self.device,
                                                 download_root=download_root)
        self.model.to(self.device)
        self.tokenizer = lambda c: clip.tokenize(f"{c}").to(self.device)
        self.logit_scale = 100.0

    @property
    def transforms(self):
        return clip_transforms(self._transforms)

    def encode_text(self, text=None):
        text = self.tokenizer(text)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features

    def _encode_image(self, image):
        embedding = self.model.encode_image(image)
        embedding /= embedding.norm(dim=1, keepdim=True)
        return embedding


class MedClip(BaseCLIP):
    def __init__(self, device, clip_name):
        from medclip import MedCLIPModel, constants, MedCLIPVisionModel, MedCLIPVisionModelViT
        self.device = device
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModel) if 'RN' in clip_name else MedCLIPModel(
            vision_cls=MedCLIPVisionModelViT)
        self.load_state_dict(clip_name)
        self.image_processor = MedCLIPFeatureExtractor()
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE, use_fast=False)

    @property
    def logit_scale(self):
        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale

    @property
    def transforms(self):
        return clip_transforms(self.image_processor)

    def load_state_dict(self, clip_name):
        '''
                If input_dir is None, download pretrained weight from google cloud and load.
                '''
        import wget
        import zipfile
        import requests
        from medclip import constants

        if 'RN' in clip_name:
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            input_dir = './pretrained/medclip-resnet'
        else:
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            input_dir = './pretrained/medclip-vit'

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

            # download url link
            pretrained_url = requests.get(pretrained_url).text
            filename = wget.download(pretrained_url, input_dir)

            # unzip
            zipf = zipfile.ZipFile(filename)
            zipf.extractall(input_dir)
            zipf.close()
            print('\n Download pretrained model from:', pretrained_url)

        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME), map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        print('load model weight from:', input_dir)

    def encode_text(self, text=None):
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True)['input_ids']
        input_ids = input_ids.to(self.device)
        text_embeds = self.model.text_model(input_ids, None)
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def _encode_image(self, image=None):
        embedding = self.model.vision_model(pixel_values=image)
        # img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return embedding


class BioMedClip(BaseCLIP):
    def __init__(self, device):
        import open_clip
        self.device = device
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model.to(self.device)
        self.logit_scale = self.model.logit_scale.exp()

    @property
    def transforms(self):
        return clip_transforms(self.preprocess_val)

    def encode_text(self, text=None):
        text = self.tokenizer(text, context_length=256).to(self.device)
        return self.model.encode_text(text, normalize=True)

    def _encode_image(self, image=None):
        embedding = self.model.encode_image(image, normalize=True)
        return embedding
