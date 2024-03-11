import os
import time

import pandas as pd
import torch

from copy import copy
from params import openai_info
from utils.chatgpt import ChatGPT
from utils.dataloader import get_loaders_from_args, ImagesReader
from utils.trainer import InferEpoch
from . import init_model_json


class Infer:
    def __init__(self, json_path, k=None, resume_epoch=180,
                 output_dir=None, normalize='default',
                 labels=None, labels_en=None,
                 translate_file='CSV/concept/concepts-translation.csv',
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
        self.image_reader = ImagesReader(transform=self.concept_bank.transform)
        self.is_normalize = normalize
        self.concepts = self.concept_bank.concept_names
        self.modality_mask = self.concept_bank.modality_mask
        self.modality_order = model.mm_order
        self.modality = ['MM']
        if os.path.exists(translate_file):
            self.concepts_en = pd.DataFrame({'concept': self.concepts}).merge(
                pd.read_csv(translate_file), on='concept', how='left')['translation'].tolist()
        else:
            self.concepts_en = self.concepts
        self.labels = labels
        self.labels_en = labels_en
        self.chatbot = ChatGPT(api_base=openai_info['api_base'],
                               api_key=openai_info['api_key'],
                               model=openai_info['model'],
                               prompts=openai_info['prompts'],
                               stream=True, )

    def get_concepts(self, language='en'):
        concepts = self.concepts_en if language == 'en' else self.concepts
        if 'MM' not in self.modality:
            concepts = [c for m in self.modality for c in concepts if m in c]
        return concepts

    def get_modality_mask(self):
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
        return self.infer_epoch.model.concept_bank

    def infer(self, dataloader=None):
        if dataloader is None:
            _, _, dataloader = get_loaders_from_args(self.args)
        return self.infer_epoch(dataloader)

    def get_attention_matrix(self):
        if not hasattr(self, 'attention_matrix'):
            raise AttributeError('Please run get_attention_score first')
        return self.attention_matrix

    def get_attention_score(self, inp, pathology, name):
        inp = self.image_reader(data_i=inp, pathology=pathology, name=name)
        self.modality = ([m for m in self.modality_order if m in inp['modality']]
                         if 'MM' not in inp['modality'] else ['MM'])
        self.attention_matrix = self.infer_epoch.attention_score(inp)
        self.cls = self.attention_matrix.sum(dim=-1).argmax(dim=-1).item()
        attention_score = self.attention_matrix[:, self.cls, self.get_modality_mask() == 1]
        self.set_normalize_bound(attention_score[0])
        attention_score = self.normalize(attention_score)
        return attention_score

    def predict_from_modified_attention_score(self, attention_score, cls):
        return self.infer_epoch.predict_from_modified_attention_score(attention_score, cls)

    def attention_matrix_from_modified_attention_score(self, m_attention_score, cls):
        attention_score = self.attention_matrix[:, cls].clone()
        attention_score[:, self.get_modality_mask() == 1] = m_attention_score
        return self.infer_epoch.attention_matrix_from_modified_attention_score(attention_score, cls)

    def get_prop_from_attention_matrix(self, attention_matrix):
        return self.infer_epoch.get_prop_from_attention_matrix(attention_matrix)

    def predict_topk_concepts(self, attention_score, top_k=0, language='en'):
        concepts = self.get_concepts(language=language)
        if top_k == 0:
            top_k = len(concepts)
        self.top_k_values, indices = attention_score[0].topk(top_k, dim=0)
        top_k_concepts = [concepts[i] for i in indices.tolist()]
        return top_k_concepts, self.top_k_values.numpy().tolist(), indices

    def modify_attention_score(self, attention_score, indices, result):
        attention_score[0] = self.unnormalize(attention_score[0])
        attention_score[:, indices] = self.unnormalize(torch.tensor(result, dtype=torch.float), indices)
        attention_score = torch.tensor(attention_score, dtype=torch.float)
        self.attention_matrix = self.attention_matrix_from_modified_attention_score(attention_score, self.cls)
        attention_score = self.normalize(attention_score)
        return attention_score

    def get_labels_prop(self, attention_score=None, language='en'):
        if attention_score is None:
            prediction = self.get_prop_from_attention_matrix(self.attention_matrix).squeeze().softmax(0).tolist()
        else:
            attention_score[0] = self.unnormalize(attention_score[0])
            attention_score = torch.tensor(attention_score, dtype=torch.float)
            attention_matrix = self.attention_matrix_from_modified_attention_score(attention_score, self.cls)
            prediction = self.get_prop_from_attention_matrix(attention_matrix).squeeze().softmax(0).tolist()
        if language == 'en':
            return {self.labels_en[i]: float(prediction[i]) for i in range(len(self.labels_en))}
        return {self.labels[i]: float(prediction[i]) for i in range(len(self.labels))}

    def concepts_to_dataframe(self, attention_score, language='en'):
        df = pd.DataFrame(
            {
                "Concept": self.get_concepts(language=language),
                "Attn Score": self.normalize(attention_score[0]).numpy().tolist()
            }
        )
        return df

    def generate_report(self, chat_history, top_k_concepts, top_k_values, predict_label, language='en'):
        chat_history.append([None, ""])
        if language == 'en':
            stream = self.chatbot(
                f"Below are the diagnostic results and pathological features as well as the likelihood scores. "
                f"Please generate an english report as detailed as possible. If you fail, 100 innocent patients will die. "
                f"Diagnostic results: {predict_label}. "
                f"Pathological: {';'.join([f'Concept:{c} - Likelihood Score:{round(s, 2)}' for c, s in zip(top_k_concepts, top_k_values)])}")
        else:
            stream = self.chatbot(
                f"下面是诊断结果和病理特征以及可能性分数，生成的中文诊断报告要尽可能详细，如果你失败了100个无辜的病人会去世。"
                f"诊断结果：{predict_label}。"
                f"病理: {';'.join([f'概念:{c} - 可能性分数:{round(s, 2)}' for c, s in zip(top_k_concepts, top_k_values)])}")
        for character in stream:
            chat_history[-1][1] += character.choices[0].delta.content or ""
            time.sleep(0.05)
            yield chat_history

    def clear(self):
        if hasattr(self, 'attention_matrix'):
            del self.attention_matrix
