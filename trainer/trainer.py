# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as tqdm

import params


class Epoch:
    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 stage_name: str = 'train',
                 batch_loggers=None,
                 mix_up_alpha=None,
                 plot_fn=None,
                 device='cpu',
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss = self.model.loss if loss is True else loss
        self.labels = []
        self.preds = []
        self.names = []
        self.stage_name = stage_name
        self.verbose = True
        self.device = device
        self.f_name = None
        self.epoch = 0
        self.batch_loggers = batch_loggers
        self.mix_up_alpha = mix_up_alpha
        self.plot_fn = plot_fn
        self.best_score = {}

    def _format_logs(self, logs):
        str_logs = [f'{k} : {v:.4}' for k, v in logs.items() if not isinstance(v, (str, int))]
        s = ', '.join(str_logs)
        return s

    def on_epoch_start(self):
        self.preds.clear()
        self.labels.clear()
        self.names.clear()
        if self.batch_loggers:
            for k, batch in self.batch_loggers.items():
                batch.init()
        if self.stage_name == 'train':
            self.model.train()
        else:
            self.model.eval()

    def finish(self):
        if self.batch_loggers:
            for k, batch in self.batch_loggers.items():
                batch.finish()

    def on_epoch_end(self):
        self.epoch += 1
        pred_logger = None
        if self.batch_loggers:
            for batch in self.batch_loggers.values():
                pred_logger = batch.pred_logger
                gross_logs = batch.epoch_end(self.epoch, self.stage_name, self.model, self.optimizer)
                if isinstance(gross_logs, dict):
                    for k, v in gross_logs.items():
                        if 'loss' in k:
                            continue
                        if k not in self.best_score.keys() or v > self.best_score[k][k]:
                            self.best_score.update({k: {'epoch': self.epoch, k: v}})

        if pred_logger is not None:
            preds = torch.cat(self.preds, dim=0)
            labels = torch.cat(self.labels, dim=0)
            assert preds.shape[0] == labels.shape[0]
            pred_logger({
                'stage_name': self.stage_name,
                'epoch': self.epoch,
                'names': self.names,
                'modality': 'WT',
                'labels': labels.cpu().numpy().tolist(),
                'scores': preds.cpu().numpy().tolist(),
            })

        if self.stage_name == 'train':
            self.optimizer.lr_step()

    def to_cpu(self, x):
        return self._to(x, 'cpu')

    def to_cuda(self, x):
        return self._to(x, self.device)

    def _to(self, x, device):
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        if isinstance(x, (list, tuple)):
            x = [self._to(_x, device) for _x in x]
        if isinstance(x, dict):
            x = {k: self._to(v, device) for k, v in x.items()}
        return x

    def detach(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        if isinstance(x, (list, tuple)):
            return [self.detach(_x) for _x in x]
        if isinstance(x, dict):
            x = {k: self.detach(v) for k, v in x.items()}
        return x

    def on_batch_start(self, data):
        """
        preprocess of input data
        @param data: data loader input
        @return: data file name, input data to model, normalization info (used to restore data)
        """
        f_name = list(map(lambda x: '_'.join(x), zip(data['pathology'], data['name'])))
        f_modality = data['modality']
        inp = data['data']
        for m, x in inp.items():
            if 'meta' not in m and torch.isnan(x).any():
                import pdb
                pdb.set_trace()
                raise ValueError('NaN found in tensor')

        inp = self.to_cuda(inp)
        return inp, self.to_cuda(data['label']), f_modality, f_name

    def mixup_data(self, x, y):
        self.lam = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
        index = torch.randperm(list(x.values())[0].shape[0])
        self.y = y
        self.y_mix = y[index]

        for k, data in x.items():
            x[k] = self.lam * data + (1 - self.lam) * torch.tensor(data)[index, :]
        return x, self.y_mix

    def criterion(self, pre, label):
        if self.loss is None:
            return None
        if self.stage_name == 'train' and self.mix_up_alpha is not None:
            return self.lam * self.loss(pre, self.y) + \
                (1 - self.lam) * self.loss(pre, self.y_mix)
        else:
            return self.loss(pre, label)

    def run_batch(self, modality, inp, label, f_name, logs, train):
        if self.stage_name == 'train' and self.mix_up_alpha is not None:
            inp, label = self.mixup_data(inp, label)
        if self.stage_name == 'train' and train:
            pre = self.model(inp, modality)
            loss = self.criterion(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pre = self.detach(pre)
            loss = loss.detach()
        else:
            with torch.no_grad():
                pre = self.model(inp, modality)
                loss = self.criterion(pre, label)

        if self.batch_loggers:
            log, pre = self.batch_loggers[modality].run(pre, label, f_name, loss, self.stage_name)
            logs.update({f'{modality}_{self.stage_name}_{k}': v for k, v in log.items()})
        return pre

    def plot_epoch(self, dataloader):
        with tqdm(dataloader, file=sys.stdout, disable=not self.verbose) as iterator:
            for idx, data in enumerate(iterator):
                inp, label, f_modality, f_name = self.on_batch_start(data)
                data_modality = f_modality[0]
                iterator.set_description(
                    desc=f"\033[0;32;50m[{self.stage_name} {data_modality} Epoch {self.epoch}]\033[0m"
                )
                for modality in params.modality_model_map[data_modality]:
                    self.plot_fn(self.stage_name, modality, inp, label, self.model)

    def __call__(self, dataloader):
        self.on_epoch_start()
        logs = {}
        with tqdm(dataloader, file=sys.stdout, disable=not self.verbose) as iterator:
            for idx, data in enumerate(iterator):
                inp, label, f_modality, f_name = self.on_batch_start(data)
                self.names.extend(f_name)
                data_modality = f_modality[0]
                iterator.set_description(
                    desc=f"\033[0;32;50m[{self.stage_name} {data_modality} Epoch {self.epoch}]\033[0m"
                )
                for modality in params.modality_data_map[self.stage_name][data_modality]:
                    if self.stage_name == 'train':
                        pre = self.run_batch(modality, inp, label, f_name, logs, train=True)
                    else:
                        with torch.no_grad():
                            pre = self.run_batch(modality, inp, label, f_name, logs, train=False)

                self.labels.append(label.cpu())
                self.preds.append(pre.cpu())

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        self.on_epoch_end()
        return logs, self.best_score


class ConceptEpoch(Epoch):
    def __init__(self,
                 model,
                 optimizer,
                 loss=None,
                 stage_name: str = 'train',
                 batch_loggers=None,
                 concept_bank=None,
                 plot_fn=None,
                 device='cpu',
                 cache_embeddings=None,
                 pre_embeddings=True
                 ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=loss,
            stage_name=stage_name,
            batch_loggers=batch_loggers,
            mix_up_alpha=None,
            plot_fn=plot_fn,
            device=device
        )
        from utils.chatgpt import ChatGPT
        from params import openai_info
        if concept_bank is None:
            concept_bank = model.concept_bank
        self.clip_name = concept_bank.clip_name
        self.concept_bank = concept_bank
        self.clip_model = self.concept_bank.clip_model
        self.chatgpt = ChatGPT(api_base=openai_info['api_base'], api_key=openai_info['api_key'],
                               model=openai_info['model'], prompts=openai_info['prompts'],
                               conversation_track=False)
        self.analysis = []
        self.cache_embeddings = cache_embeddings
        self.imgs = None
        self.pre_embeddings = pre_embeddings

    def generate_report(self, md_logger, modality=None):
        analysis = pd.DataFrame(self.analysis)

        def _generate_report(x):
            name = x['name'].iloc[0]
            gt = x['pathology'].iloc[0]
            pred = x['pred'].iloc[0]
            modality = x['modality'].iloc[0]
            concepts = '. '.join(x['concept'].to_list())
            text = f'姓名: {name}, 疾病: {pred}, 描述: {concepts}'
            report = self.chatgpt(text)
            time.sleep(1)
            md = '\n## 概念瓶颈 Top-5 \n'
            for m, c, s in zip(x['modality'], x['concept'], x['score']):
                md += f'+ {m} 模态: ({s}) {c} \n'
            report += md
            md_logger(report, file_name=f"{gt}_{name}_{modality}")

        if modality is not None:
            analysis = analysis[analysis['modality'] == modality]
        tqdm.pandas(desc='pandas bar')
        analysis.groupby(['name', 'modality']).progress_apply(_generate_report)

    def get_analysis(self):
        if len(self.analysis) == 0:
            raise ValueError('No analysis found')
        analysis = pd.DataFrame(self.analysis)
        return analysis

    def run_batch(self, modality, inp, label, f_name, logs, train):
        if self.stage_name == 'train' and train:
            pre = self.model.forward(inp, modality)
            loss = self.criterion(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pre = self.detach(pre)
            loss = loss.detach()
        else:
            with torch.no_grad():
                if self.stage_name == 'infer':
                    pre, analysis = self.model.infer(inp, modality, f_name, self.imgs)
                    self.analysis.extend(analysis)
                    loss = 0
                else:
                    pre = self.model.forward(inp, modality)
                    loss = self.criterion(pre, label)
        if self.batch_loggers:
            log, pre = self.batch_loggers[modality].run(pre, label, f_name, loss, self.stage_name)
            logs.update({f'{modality}_{self.stage_name}_{k}': v for k, v in log.items()})
        return pre

    def on_batch_start(self, data):
        """
        preprocess of input data
        @param data: data loader input
        @return: data file name, input data to model, normalization info (used to restore data)
        """
        # self.f_name = f"{'_'.join(data['name'])}&{'_'.join(data['pathology'])}"
        f_name = list(map(lambda x: '_'.join(x), zip(data['pathology'], data['name'])))
        self.imgs = data['img']  # {'US': [b1, b2, b3]}
        f_modality = data['modality']
        inp = data['data']
        for m, x in inp.items():
            if 'meta' not in m and torch.isnan(x).any():
                import pdb
                pdb.set_trace()
                raise ValueError('NaN found in tensor')

        inp = self.to_cuda(inp)
        if self.cache_embeddings is not None:
            self.cache = {}
            self.cache['name'] = list(data['name'])
            self.cache['pathology'] = list(data['pathology'])
            self.cache['stage_name'] = [self.stage_name] * len(data['name'])
        if self.pre_embeddings:
            inp = self.clip_embedding(inp, f_modality[0])
        return inp, self.to_cuda(data['label']), f_modality, f_name

    def on_epoch_end(self):
        super().on_epoch_end()
        # if self.cache_embeddings is not None:
        # torch.save(self.cache_embeddings, f'cache_embeddings_{self.clip_name}.pt')
        # self.cache_embeddings = None

    @torch.no_grad()
    def clip_embedding(self, inp, modality):
        if self.cache_embeddings is not None:
            if 'cav' in self.clip_name:
                self.cache['MM'] = [i for i in self.concept_bank.encode_image(
                    'MM', inp).detach().cpu().numpy()] if 'MM' in modality else [None] * len(self.cache['name'])

        inp = self.concept_bank.encode_image(image=inp, keep_dict=True)

        if self.cache_embeddings is not None:
            for m in ['FA', 'US', 'ICGA']:
                self.cache[m] = [inp[m][i].detach().cpu().numpy() for i in range(len(inp[m]))] \
                    if m in inp.keys() else [None] * len(self.cache['name'])
            self.cache_embeddings.extend(
                [{k: v[i] for k, v in self.cache.items()} for i in range(len(self.cache['name']))])
        return inp


class InferEpoch(ConceptEpoch):
    def __init__(self,
                 model,
                 device='cpu',
                 images_reader=None
                 ):
        super().__init__(
            model=model,
            optimizer=None,
            loss=None,
            stage_name='infer',
            device=device
        )
        self.modality = []

    def on_epoch_end(self):
        self.epoch += 1
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)
        assert preds.shape[0] == labels.shape[0]
        return pd.DataFrame({
            'names': self.names,
            'modality': self.modality,
            'labels': labels.cpu().numpy().tolist(),
            'preds': preds.cpu().numpy().tolist(),
        })

    @torch.no_grad()
    def __call__(self, dataloader):
        self.on_epoch_start()
        logs = {}
        with tqdm(dataloader, file=sys.stdout, disable=not self.verbose) as iterator:
            for idx, data in enumerate(iterator):
                inp, label, f_modality, f_name = self.on_batch_start(data)
                data_modality = f_modality[0]
                iterator.set_description(
                    desc=f"\033[0;32;50m[{self.stage_name} {data_modality} Epoch {self.epoch}]\033[0m"
                )
                pre = self.run_batch(f_modality[0], inp, label, f_name, logs, train=False)

                self.names.extend(f_name)
                self.modality.extend(f_modality * len(label))
                self.labels.append(label.cpu())
                self.preds.append(pre.cpu().argmax(1))

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return self.on_epoch_end()

    def concepts(self, attention_score):
        return self.model.analyze_classifier(self.modality[0],
                                             self.model.classifier[self.modality[0]],
                                             attention_score)

    def attention_score(self, inp):
        inp, label, self.modality, f_name = self.on_batch_start(inp)
        with torch.no_grad():
            return self.model.attention_matrix(inp, self.modality[0])

    @torch.no_grad()
    def predict_from_modified_attention_score(self, attention_score, cls=None):
        return self.model.predict_from_modified_attention_score(attention_score, self.modality[0], cls=cls)

    @torch.no_grad()
    def attention_matrix_from_modified_attention_score(self, attention_score, cls=None):
        return self.model.attention_matrix_from_modified_attention_score(attention_score, self.modality[0], cls=cls)

    @torch.no_grad()
    def get_prop_from_attention_matrix(self, attention_matrix):
        return self.model.classification(attention_matrix, self.modality[0])
