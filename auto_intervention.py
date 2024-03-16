import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.data_split import DataSplit
from run_analysis import get_concepts_gt
from inference import Infer
from params import pathology_labels_cn_to_en, data_info, pathology_labels
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class AutoIntervention:
    def __init__(self, bias_max=0.0, bias_min=0.0, on_error=True, fold=0, root_dir='Analysis/intervention',
                 device='cpu', json_path=None):
        self.attention_score = None
        self.min = None
        self.max = None
        self.bias_max = bias_max
        self.bias_min = bias_min
        self.on_error = on_error
        self.root_dir = f'{root_dir}_biasmax{bias_max}biasmin{bias_min}/fold{fold}'
        if json_path is None:
            json_path = f'result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_add_aow_zero_MM_max/fold_{fold}_report_strict_r1.0_c1.0'
        os.makedirs(self.root_dir, exist_ok=True)
        self.predictor = Infer(
            json_path=json_path,
            device=device,
            labels=list(pathology_labels_cn_to_en.keys()),
            labels_en=list(pathology_labels_cn_to_en.values()),
            normalize='default',
            backbone=f'Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_{fold}',
            language='cn',
        )
        self.concepts_gt = get_concepts_gt()
        self.concepts_gt['concept'] = self.concepts_gt['time'] + self.concepts_gt['concept']
        self.concepts_gt['concept'] = self.concepts_gt['modality'] + ', ' + self.concepts_gt['concept']
        self.test_split = DataSplit(data_path=data_info['data_path'],
                                    csv_path=data_info['csv_path']).get_test_data()
        if self.on_error:
            self.error_pred: pd.DataFrame = pd.read_csv(
                'result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_add_aow_zero_MM_max/'
                f'fold_{fold}_report_strict_r1.0_c1.0/CSVLogger/error_name.csv')
            self.error_pred: pd.DataFrame = self.error_pred.loc[(self.error_pred['epoch'] == 180) &
                                                                (self.error_pred['modality'] != 'WT') &
                                                                (self.error_pred['stage_name'] == 'test')].drop(
                columns=['epoch', 'stage_name']
            )
            self.error_pred['pathology'] = self.error_pred['name'].map(lambda x: x.split('_')[0])
            self.error_pred['name'] = self.error_pred['name'].map(lambda x: x.split('_')[-1])
            assert all(self.error_pred['name'].isin(self.concepts_gt['name'].unique()))
        assert all(self.test_split['name'].isin(self.concepts_gt['name'].unique()))
        self.k = [[5, 0],
                  [10, 0],
                  [15, 0],
                  [20, 0],
                  [0, 5],
                  [0, 10],
                  [0, 15],
                  [0, 20],
                  [5, 5],
                  [10, 10],
                  [15, 15],
                  [20, 20]]
        self.strategies = ['max-min', 'max-None', 'max-half', 'max-zero', 'None-None',
                           'max-random', 'random-random']

    def get_test_data(self, names=None, modality='MM'):
        if names is None:
            names = self.error_pred['name'].unique()
        if modality is None:
            modality = ['FA', 'ICGA', 'US']
        if isinstance(names, str):
            names = [names]
        test = self.test_split[self.test_split['name'].isin(names)]

        def _fn(x):
            path = [x['name'], x['pathology']]
            for m in ['FA', 'ICGA', 'US']:
                if m in modality or modality == 'MM':
                    path.extend(x['path'][m])
                elif m == 'US':
                    path.append(None)
                else:
                    path.extend([None, None, None])
            return path

        test = test.apply(_fn, axis=1).to_list()
        return test

    def get_concepts(self, name, modality):
        concepts = self.concepts_gt[(self.concepts_gt['name'] == name) & (self.concepts_gt['modality'] == modality)]
        return concepts['concept'].values

    def predict_concepts(self, name, pathology, imgs):
        inp = dict(FA=imgs[:3], ICGA=imgs[3:6], US=imgs[6:])
        attention_score = self.predictor.get_attention_score(inp=inp, name=name,
                                                             pathology=pathology)
        concepts_tuple = self.predictor.predict_topk_concepts(
            attention_score,
            top_k=0,  # all concepts
        )
        self.max = attention_score.max().cpu().item()
        self.min = attention_score.min().cpu().item()
        self.max = self.max - self.bias_max * (self.max - self.min)
        self.min = self.min + self.bias_min * (self.max - self.min)
        self.attention_score = attention_score
        return concepts_tuple

    def operator(self, op, v):
        """
            max: 最大值
            min: 最小值
            None: 不干预
        """
        if op == 'max':
            return self.max
        if op == 'min':
            return self.min
        if op == 'half':
            return v / 2
        if op == 'zero':
            return 0

        if op == 'random':
            return random.random() * (self.max - self.min) + self.min
        if op == 'None':
            return None
        raise NotImplementedError

    def concept_exits(self, concept, concepts_gts):
        # ignore time
        m = concept.split(', ')[0]
        concept = concept.split(', ')[1]
        concept = (concept if '期' not in concept and len(concept.split('期')) != 2 else concept.split('期')[-1])
        for concepts_gt in concepts_gts:
            if concept in concepts_gt:
                return True
        return False

    def modify_strategy(self, strategy, concepts_gt, concepts_tuple):
        """
            add-sub: 命中 add，未命中 sub
            add-random: 命中 add，未命中 random
        """
        values, indices = [], []
        for idx, (c, s, i) in enumerate(zip(*concepts_tuple)):
            i = i.cpu().item()
            in_op, nin_op = strategy.split('-')
            if self.concept_exits(c, concepts_gt):
                value = self.operator(in_op, s)
            else:
                value = self.operator(nin_op, s)
            if value is None:
                continue
            indices.append(i)
            values.append(value)
        return values, indices

    def modify_attn_score(self, name, modality, concepts_tuple, top_k=5, bottom_k=5,
                          top_strategy='strict', bottom_strategy='strict'):
        if modality == 'MM':
            concepts_gt = self.concepts_gt[self.concepts_gt['name'] == name]['concept'].to_list()
        else:
            concepts_gt = self.concepts_gt[(self.concepts_gt['name'] == name) &
                                           (self.concepts_gt['modality'] == modality)]['concept'].to_list()
        if top_k > 0:
            top_values, top_indices = self.modify_strategy(top_strategy,
                                                           concepts_gt=concepts_gt,
                                                           concepts_tuple=[t[:top_k] for t in concepts_tuple])
        else:
            top_values, top_indices = [], []
        if bottom_k > 0:
            bottom_values, bottom_indices = self.modify_strategy(bottom_strategy,
                                                                 concepts_gt=concepts_gt,
                                                                 concepts_tuple=[b[-bottom_k:] for b in
                                                                                 concepts_tuple])
        else:
            bottom_values, bottom_indices = [], []
        attention_score = self.predictor.modify_attention_score(self.attention_score.clone(),
                                                                top_indices + bottom_indices,
                                                                top_values + bottom_values,
                                                                inplace=False)
        return attention_score

    def predict_from_attn_score(self, attention_score):
        labels = self.predictor.get_labels_prop(attention_score, class_type='int')
        return max(labels, key=lambda k: labels[k])

    def intervention_iter(self, name, pathology, modality, label):
        results_melt = []
        print(f'Processing {name} {pathology} {modality}')
        result = {
            'name': name,
            'pathology': pathology,
            'modality': modality,
            'label': label,
        }
        imgs = self.get_test_data(names=name, modality=modality)[0][2:]
        concepts_tuple = self.predict_concepts(name, pathology, imgs)
        for top_k, bottom_k in self.k:
            if (
                    ((top_k + bottom_k) > 26 and modality == 'US') or
                    ((top_k + bottom_k) > 30 and modality == 'ICGA')

            ):
                continue

            for strategy in self.strategies:
                attention_score = self.modify_attn_score(name, modality, concepts_tuple,
                                                         top_k=top_k, bottom_k=bottom_k,
                                                         top_strategy=strategy,
                                                         bottom_strategy=strategy, )
                pred = self.predict_from_attn_score(attention_score)
                result[f'T{top_k} B{bottom_k} {strategy}'] = pred
                results_melt.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': modality,
                    'label': label,
                    'intervention': f'T{top_k} B{bottom_k}',
                    'strategy': strategy,
                    'pred': pred,
                })
        return [result], results_melt

    def intervention_on_error(self, force=False):
        if os.path.exists(os.path.join(self.root_dir, 'error_auto_intervention_predict.csv')) and not force:
            return pd.read_csv(os.path.join(self.root_dir, 'error_auto_intervention_predict.csv'))
        results = []
        results_melt = []
        for i, row in self.error_pred[['name', 'pathology', 'label', 'modality']].iterrows():
            name, pathology, modality, label = row['name'], row['pathology'], row['modality'], row['label']
            result, results_melt = self.intervention_iter(name, pathology, modality, label)
            results.extend(result)
            results_melt.extend(results_melt)
        results_melt = pd.DataFrame(results_melt)
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(self.root_dir, 'error_auto_intervention_predict.csv'), index=False)
        results_melt.to_csv(os.path.join(self.root_dir, 'error_auto_melt_intervention_predict.csv'), index=False)
        return results

    def intervention_on_test(self, force=False):
        if os.path.exists(os.path.join(self.root_dir, 'test_auto_intervention_predict.csv')) and not force:
            return pd.read_csv(os.path.join(self.root_dir, 'test_auto_intervention_predict.csv'))
        results = []
        results_melt = []
        test = self.test_split[['name', 'pathology']].drop_duplicates()
        test['label'] = test['pathology'].map(lambda x: pathology_labels[x])
        for i, row in test.iterrows():
            name, pathology, label = row['name'], row['pathology'], row['label']
            for modality in ['FA', 'ICGA', 'US', 'MM']:
                result, results_melt = self.intervention_iter(name, pathology, modality, label)
                results.extend(result)
                results_melt.extend(results_melt)
        results_melt = pd.DataFrame(results_melt)
        results = pd.DataFrame(results)
        results.to_csv(
            os.path.join(self.root_dir, 'test_auto_intervention_predict.csv'),
            index=False)
        results_melt.to_csv(
            os.path.join(self.root_dir, 'test_auto_melt_intervention_predict.csv'),
            index=False)
        return results

    def compute_metrics(self, df, file_name):
        if 'pathology' in df.columns and 'name' in df.columns:
            df = df.drop(columns=['name', 'pathology'])
        FA = df[df['modality'] == 'FA'].drop(columns=['modality']).dropna(axis='columns', how='any')
        ICGA = df[df['modality'] == 'ICGA'].drop(columns=['modality']).dropna(axis='columns', how='any')
        US = df[df['modality'] == 'US'].drop(columns=['modality']).dropna(axis='columns', how='any')
        MM = df[df['modality'] == 'MM'].drop(columns=['modality']).dropna(axis='columns', how='any')
        fa = self._metric(FA, 'FA')
        icga = self._metric(ICGA, 'ICGA')
        us = self._metric(US, 'US')
        mm = self._metric(MM, 'MM')
        df = pd.concat([fa, icga, us, mm])
        df.to_csv(os.path.join(self.root_dir, file_name), index=False)

    def _metric(self, df, modality):
        data = df.to_dict(orient='list')
        label = data.pop('label')
        metric = []
        for key, value in data.items():
            metric.append({
                'intervention': '+'.join(key.split(' ')[:-1]),
                'strategy': key.split(' ')[-1],
                'modality': modality,
                'f1 score': f1_score(label, value, average='macro'),
                'accuracy': accuracy_score(label, value),
                'recall': recall_score(label, value, average='macro'),
                'precision': precision_score(label, value, average='macro')
            })
        metric = pd.DataFrame(metric).melt(id_vars=['modality', 'intervention', 'strategy'],
                                           value_vars=['f1 score', 'accuracy', 'recall', 'precision'],
                                           var_name='metrics', value_name='score')
        return metric

    def error_expand(self):
        test_data = {
            0: 11,
            1: 8,
            2: 39
        }
        intervention = pd.read_csv(os.path.join(self.root_dir, 'error_auto_intervention_predict.csv'))
        intervention = intervention.drop(columns=['name', 'pathology'])
        modalities = intervention['modality'].unique()
        results = []
        for modality in modalities:
            cls_d = (intervention[intervention['modality'] == modality].drop(columns=['modality'])
                     .dropna(axis='columns', how='any').to_dict(orient='list'))
            inter_cls = cls_d['label']
            cls_list = []
            for cls, num in test_data.items():
                inter_cls_num = inter_cls.count(cls)
                cls_list.extend([cls] * (num - inter_cls_num))
            for key, value in cls_d.items():
                cls_d[key] = value + cls_list
            cls_df = pd.DataFrame(cls_d)
            cls_df['modality'] = modality
            results.append(cls_df)
        results = pd.concat(results)
        return results

    def long_to_wide(self, filename):
        os.makedirs(os.path.join(self.root_dir, f'wide_{filename.split(".")[0]}'), exist_ok=True)
        df = pd.read_csv(os.path.join(self.root_dir, filename))
        for metric in df['metrics'].unique():
            metric_df = df[df['metrics'] == metric].drop(columns=['metrics'])
            for strategy in metric_df['strategy'].unique():
                dfs = []
                strategy_df = metric_df[metric_df['strategy'] == strategy].drop(columns=['strategy'])
                for modality in strategy_df['modality'].unique():
                    df_dict = {'modality': modality}
                    modality_df = strategy_df[strategy_df['modality'] == modality].drop(columns=['modality'])
                    for intervention, score in zip(strategy_df['intervention'], modality_df['score']):
                        df_dict[intervention] = score
                    dfs.append(df_dict)
                dfs = pd.DataFrame(dfs)
                dfs = dfs.set_index('modality')
                dfs.to_csv(os.path.join(self.root_dir, f'wide_{filename.split(".")[0]}', f'{strategy}_{metric}.csv'),
                           index=True)

    def model_focused_concepts(self):
        results = []
        for cls, topks in self.predictor.focused_concepts_from_cls(0, 'MM').items():
            results.extend([{
                'pathology': cls,
                'concept': topk[0],
                'score': topk[1],
            } for topk in topks])
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(self.root_dir, 'model_focused_concepts.csv'), index=False)
        plt.figure(figsize=(20, 5), dpi=200)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        sns.barplot(data=results, x='concept', y='score', hue='pathology')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_dir, 'cls_focused_concepts.png'))

    def human_focused_concepts_from_testdata(self):
        concepts_gt = self.concepts_gt[['pathology', 'concept', 'name']].drop_duplicates()
        concepts_gt['count'] = concepts_gt.groupby(['pathology', 'concept'])['name'].transform('count').drop(
            columns=['name'])
        concepts_gt = concepts_gt.drop_duplicates()
        concepts_order = self.predictor.concepts
        concepts_gt = concepts_gt[concepts_gt['concept'].isin(concepts_order)]
        concepts_gt['sort_key'] = concepts_gt['concept'].map(lambda x: concepts_order.index(x))
        concepts_gt = concepts_gt.sort_values('sort_key').drop('sort_key', axis=1)

        def _fn(x):
            x['count'] = x['count'].map(
                lambda c: (c - x['count'].min()) / (x['count'].max() - x['count'].min())
            )
            return x

        concepts_gt = concepts_gt.groupby('pathology').apply(_fn).reset_index(drop=True)
        concepts_gt.to_csv(os.path.join(self.root_dir, 'human_focused_concepts.csv'), index=False)
        plt.figure(figsize=(20, 5), dpi=200)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        sns.barplot(data=concepts_gt, x='concept', y='count', hue='pathology')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_dir, 'human_focused_concepts.png'))


def intervene(bias_min=0.0, fold=0, on_error=True, device='cuda:0', json_path=None, root_dir='Analysis/intervention'):
    auto_intervention = AutoIntervention(bias_min=bias_min, device=device, on_error=on_error,
                                         fold=fold, root_dir=root_dir, json_path=json_path)
    results = auto_intervention.intervention_on_test(force=True)
    auto_intervention.compute_metrics(results, 'test_intervention_metrics.csv')

    results = auto_intervention.intervention_on_error(force=True)
    expand = auto_intervention.error_expand()
    auto_intervention.compute_metrics(results, 'local_error_intervention_metrics.csv')
    auto_intervention.compute_metrics(expand, 'expand_error_intervention_metrics.csv')

    auto_intervention.long_to_wide('local_error_intervention_metrics.csv')
    auto_intervention.long_to_wide('expand_error_intervention_metrics.csv')
    auto_intervention.long_to_wide('test_intervention_metrics.csv')


def five_fold():
    for fold in range(5):
        intervene(fold=fold, on_error=True)


def bias_min():
    for i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
        intervene(bias_min=i, fold=0, on_error=True)


def human():
    for fold in range(5):
        intervene(
            json_path=f'result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_human_clean_strict_add_aow_zero_MM_max/fold_{fold}_human_clean_strict_r1.0_c1.0',
            on_error=True,
            root_dir='Analysis/intervention_human',
            fold=fold
        )


def focused_concepts():
    auto_intervention = AutoIntervention(bias_min=0.0, device='cuda:0', on_error=True,
                                         fold=0, root_dir='Analysis/intervention')
    # auto_intervention.model_focused_concepts()
    auto_intervention.human_focused_concepts_from_testdata()


if __name__ == "__main__":
    focused_concepts()
