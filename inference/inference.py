# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os
import time

import pandas as pd
import torch

from copy import copy
from params import openai_info, pathology_labels_cn_to_en
from utils.chatgpt import ChatGPT
from utils.dataloader import get_loaders_from_args, ImagesReader
from trainer.trainer import InferEpoch
from . import init_model_json


class Infer:
    def __init__(self, json_path, k=None, resume_epoch=180,
                 output_dir=None, normalize='default', language='en',
                 **kwargs):
        super().__init__()
        model, args = init_model_json(json_path, k, resume_epoch, **kwargs)
        self.args = args
        self.save_path = os.path.join(output_dir, args.dir_name, 'Inference') if output_dir is not None \
            else f'{args.output_dir}/{args.dir_name}/Inference'
        os.makedirs(self.save_path, exist_ok=True)
        self.infer_epoch = InferEpoch(
            model=model,
            device=args.device,
        )
        self.image_reader = ImagesReader(
            transform=self.concept_bank.transform if self.concept_bank is not None
            else self.infer_epoch.model.transform)
        self.is_normalize = normalize
        self.modality_mask = self.concept_bank.modality_mask if self.concept_bank is not None else None
        self.modality_order = model.mm_order
        self.modality = ['MM']
        self.labels = list(pathology_labels_cn_to_en.keys())
        self.labels_en = list(pathology_labels_cn_to_en.values())
        self.language = language
        self.chatbot = ChatGPT(api_base=openai_info['api_base'],
                               api_key=openai_info['api_key'],
                               model=openai_info['model'],
                               prompts=openai_info['prompts'],
                               stream=True, )

    def raise_noninteract_error(self):
        if not hasattr(self.infer_epoch.model, 'attention_matrix'):
            raise AttributeError('Current model does not support interactive inference')

    def set_language(self, language):
        self.language = language

    def set_random(self, random):
        self.random = random

    def get_random(self):
        return self.random

    def get_concepts(self, modality=None):
        self.raise_noninteract_error()
        if modality is None:
            modality = self.modality
        if isinstance(modality, str):
            modality = [modality]
        self.concept_bank.language = self.language
        concepts = self.concept_bank['_'.join(modality)].concept_names
        if 'MM' not in modality:
            concepts = [c for m in modality for c in concepts if m in c]
        return concepts

    def get_modality_mask(self):
        self.raise_noninteract_error()
        if 'MM' in self.modality or (len(self.modality) == 1 and 'US' in self.modality):
            return torch.ones(self.concept_bank[self.modality[0]].n_concepts)
        modality = copy(self.modality)
        if 'FA' in modality and 'ICGA' in modality:
            modality.remove('ICGA')
        mask = torch.zeros(sum([self.concept_bank[m].n_concepts for m in modality]))
        for m in self.modality:
            mask[torch.where(self.modality_mask[m])[0]] = 1
        return mask

    def set_normalize_bound(self, x):
        self.min_ = x.min()
        self.max_ = x.max()
        self.neg = x < 0

    def normalize(self, x):
        if self.is_normalize == 'linear':
            return (x - self.min_) / (self.max_ - self.min_)
        elif self.is_normalize == 'abs':
            return x.abs()
        elif self.is_normalize == 'default':
            return x

    def unnormalize(self, x, indices=None):
        if self.is_normalize == 'linear':
            return x * (self.max_ - self.min_) + self.min_
        elif self.is_normalize == 'abs':
            if indices is not None:
                x[self.neg[indices]] = -x[self.neg[indices]].abs()
            else:
                x[self.neg] = -x[self.neg].abs()
            return x
        elif self.is_normalize == 'default':
            return x

    @property
    def concept_bank(self):
        if hasattr(self.infer_epoch.model, 'concept_bank'):
            return self.infer_epoch.model.concept_bank

    def infer(self, dataloader=None):
        if dataloader is None:
            _, _, dataloader = get_loaders_from_args(self.args)
        return self.infer_epoch(dataloader)

    def get_attention_matrix(self, inp=None, name=None):
        if inp is not None:
            inp = self.image_reader(data_i=inp, pathology=None, name=name)
            self.modality = ([m for m in self.modality_order if m in inp['modality']]
                             if 'MM' not in inp['modality'] else ['MM'])
            self.set_attention_matrix(self.infer_epoch.attention_score(inp))
        self.raise_noninteract_error()
        if not hasattr(self, 'attention_matrix'):
            raise AttributeError('Please run get_attention_score first')
        return self.attention_matrix

    def set_attention_matrix(self, attention_matrix):
        self.attention_matrix = attention_matrix

    def get_attention_score(self, inp=None, name=None):
        self.raise_noninteract_error()
        self.get_attention_matrix(inp, name)
        if self.random:
            self.attention_matrix = torch.rand_like(self.attention_matrix)
        self.cls = self.attention_matrix.sum(dim=-1).argmax(dim=-1).item()
        attention_score = self.attention_matrix[:, self.cls, self.get_modality_mask() == 1]
        self.set_normalize_bound(attention_score[0])
        attention_score = self.normalize(attention_score)
        return attention_score

    def predict_from_modified_attention_score(self, attention_score, cls):
        self.raise_noninteract_error()
        return self.infer_epoch.predict_from_modified_attention_score(attention_score, cls)

    def attention_matrix_from_modified_attention_score(self, m_attention_score, cls):
        self.raise_noninteract_error()
        attention_score = self.attention_matrix[:, cls].clone()
        attention_score[:, self.get_modality_mask() == 1] = m_attention_score
        return self.infer_epoch.attention_matrix_from_modified_attention_score(attention_score, cls)

    def get_prop_from_attention_matrix(self, attention_matrix):
        self.raise_noninteract_error()
        return self.infer_epoch.get_prop_from_attention_matrix(attention_matrix)

    def predict_topk_concepts(self, attention_score, top_k=0, language=None):
        self.raise_noninteract_error()
        if language is not None:
            self.set_language(language)
        concepts = self.get_concepts()
        if top_k == 0:
            top_k = len(concepts)
        self.top_k_values, indices = attention_score[0].topk(top_k, dim=0)
        top_k_concepts = [concepts[i] for i in indices.tolist()]
        return top_k_concepts, self.top_k_values.cpu().numpy().tolist(), indices

    def modify_attention_score(self, attention_score, indices, result, inplace=True):
        self.raise_noninteract_error()
        attention_score = attention_score.clone()
        attention_score[0] = self.unnormalize(attention_score[0])
        attention_score[:, indices] = self.unnormalize(
            torch.tensor(result, dtype=torch.float, device=attention_score.device), indices)
        attention_score = torch.tensor(attention_score, dtype=torch.float)
        if inplace:
            self.attention_matrix = self.attention_matrix_from_modified_attention_score(attention_score, self.cls)
        attention_score = self.normalize(attention_score)
        return attention_score

    def get_labels_prop(self, attention_score=None, class_type='str', language=None, inp=None, random=True):
        if language is not None:
            self.set_language(language)
        if inp is not None:
            inp = self.image_reader(data_i=inp, pathology=None)
            self.modality = ([m for m in self.modality_order if m in inp['modality']]
                             if 'MM' not in inp['modality'] else ['MM'])
            prediction = self.infer_epoch.inference(inp).squeeze()
        elif attention_score is None:
            prediction = self.get_prop_from_attention_matrix(self.attention_matrix).squeeze()
        else:
            attention_score = attention_score.clone()
            attention_score[0] = self.unnormalize(attention_score[0])
            attention_score = torch.tensor(attention_score, dtype=torch.float)
            attention_matrix = self.attention_matrix_from_modified_attention_score(attention_score, self.cls)
            prediction = self.get_prop_from_attention_matrix(attention_matrix).squeeze()
        if self.random and random:
            prediction = torch.rand_like(prediction)
        prediction = prediction.softmax(0).tolist()
        if self.language == 'en':
            return {self.labels_en[i] if class_type == 'str' else i: float(prediction[i]) for i in
                    range(len(self.labels_en))}
        return {self.labels[i] if class_type == 'str' else i: float(prediction[i]) for i in range(len(self.labels))}

    def concepts_to_dataframe(self, attention_score, language=None):
        self.raise_noninteract_error()
        if language is not None:
            self.set_language(language)
        df = pd.DataFrame(
            {
                "Concept": self.get_concepts(),
                "Concept Activation Score": attention_score[0].numpy().tolist()
            }
        )
        return df

    def generate_report(self, chat_history, top_k_concepts, top_k_values, predict_label, language=None):
        self.raise_noninteract_error()
        if language is not None:
            self.set_language(language)
        chat_history.append([None, ""])
        if self.language == 'en':
            stream = self.chatbot(
                f"Below are the diagnostic results and pathological features as well as the likelihood scores. "
                f"Please generate an english report as detailed as possible. "
                f"Diagnostic results: {predict_label}. "
                f"Pathological: {';'.join([f'Concept:{c} - Likelihood Score:{round(s, 2)}' for c, s in zip(top_k_concepts, top_k_values)])}")
        else:
            stream = self.chatbot(
                f"下面是诊断结果和病理特征以及可能性分数，生成的中文诊断报告要尽可能详细。"
                f"诊断结果：{predict_label}。"
                f"病理: {';'.join([f'概念:{c} - 可能性分数:{round(s, 2)}' for c, s in zip(top_k_concepts, top_k_values)])}")
        for character in stream:
            chat_history[-1][1] += character.choices[0].delta.content or ""
            time.sleep(0.05)
            yield chat_history

    def clear(self):
        if hasattr(self, 'attention_matrix'):
            del self.attention_matrix

    def focused_concepts_from_cls(self, topk, modality=None, language=None):
        if language is not None:
            self.set_language(language)
        self.modality = [modality]
        concepts = self.get_concepts(modality)
        labels = self.labels_en if self.language == 'en' else self.labels
        weight = torch.sigmoid(self.infer_epoch.model.classifier[modality].weight)[..., self.get_modality_mask() == 1]
        if topk == 0:
            indices = torch.IntTensor([list(range(len(concepts)))] * 3)
            values = weight
        else:
            indices = torch.topk(weight, topk).indices
            values = torch.topk(weight, topk).values
        return {labels[i]: [[concepts[j], v] for j, v in zip(indice, value)] for i, (indice, value) in
                enumerate(zip(indices.tolist(), values.tolist()))}

    def grad_cam(self, inp, modality):
        import numpy as np
        import cv2
        from visualize.utils import compute_gradcam
        inp = self.image_reader(data_i=inp, pathology=None)['data']
        cam = compute_gradcam(self.infer_epoch.model, inp, modality).cpu().numpy()
        inp = inp[modality].cpu().numpy()[0]
        blue_lower_bound = np.array([128, 0, 0])  # 蓝色的低阈值
        blue_upper_bound = np.array([255, 50, 50])  # 蓝色的高阈值
        superimposed_img = []
        for i, ca in zip(inp, cam):
            i = np.transpose(np.array((i - i.min()) / (i.max() - i.min()) * 255, dtype=np.uint8), (1, 2, 0))
            if ca.max() != ca.min():
                ca = 1 - np.transpose(ca, (1, 2, 0))
                ca = np.array(ca * 255, dtype=np.uint8)
                ca = cv2.applyColorMap(ca, cv2.COLORMAP_JET)
                # 反转掩码：非蓝色区域为1，蓝色区域为0
                mask = cv2.bitwise_not(cv2.inRange(ca, blue_lower_bound, blue_upper_bound))
                red_map = ca * (np.expand_dims(mask, -1) > 0)
                red_map = cv2.addWeighted(red_map, .3, i, .7, 0.)
            else:
                red_map = i
            superimposed_img.append(red_map)
        return superimposed_img
