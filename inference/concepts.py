import os
from utils.dataloader import get_loaders_from_args
from utils.trainer import ConceptEpoch
from utils.decorator import cache_df


def get_topk(df, column='modality', k=None, concept_num_by_modality=None):
    if concept_num_by_modality is None:
        concept_num_by_modality = {'FA': 3,
                                   'ICGA': 2,
                                   'US': 5,
                                   'MM': 10}

    def select_topk(x):
        return x.sort_values(by='score', ascending=False)[
               :concept_num_by_modality[x[column].iloc[0]] if k is None else k]

    if k is None:
        output = df.groupby(['name', 'pathology', column]).apply(select_topk).reset_index(drop=True).sort_values(
            by=['name', 'score'], ascending=[True, False])
    else:
        output = df.groupby(['name', 'pathology']).apply(select_topk).reset_index(drop=True).sort_values(
            by=['name', 'score'], ascending=[True, False])
    return output


class ConceptInfer:
    cache = True

    def __init__(self, args, model, output_dir=None, concept_num_by_modality=None, dataloader=None, cache=True):
        super().__init__()
        self.args = args
        self.save_path = os.path.join(output_dir, args.dir_name, 'ConceptInfer') if output_dir is not None \
            else f'{args.output_dir}/{args.dir_name}/ConceptInfer'
        os.makedirs(self.save_path, exist_ok=True)
        self.get_topk = lambda df, column='modality', k=None: get_topk(df, column, k, concept_num_by_modality)
        if dataloader is None:
            _, _, self.dataloader = get_loaders_from_args(args)
        else:
            self.dataloader = dataloader
        self._cache_embeddings = []
        self.test_epoch = ConceptEpoch(
            model=model,
            loss=None,
            optimizer=None,
            stage_name='infer',
            device=args.device,
            batch_loggers=None,
            concept_bank=args.concept_bank,
            cache_embeddings=self._cache_embeddings
        )
        self.intermediate_score = []

        def hook(x):
            self.intermediate_score.append(x)

        model.remove_hook()
        model.register_hook(hook)
        self.cache = cache

    @property
    def cache_embeddings(self):
        self.infer()
        return self._cache_embeddings

    def infer(self):
        if not hasattr(self, 'concept'):
            self.test_epoch(self.dataloader)
            self.concept = self.test_epoch.get_analysis()
            self.concept.loc[:, 'score'] = self.concept['score'].map(float)
            self.concept['time'] = self.concept['concept'].map(
                lambda x: '无' if '期' not in x else ('晚期' if '晚' in x else '早期'))

    @cache_df(cache=cache)
    def _format_concepts(self, name):
        self.infer()
        if name == 'single':
            concepts = self.concept[(self.concept['img'] == '') & (self.concept['modality'] != 'MM')].drop_duplicates()
        elif name == 'img':
            concepts = self.concept[(self.concept['img'] != '') & (self.concept['modality'] != 'MM')].drop_duplicates()
        elif name == 'mm':
            concepts = self.concept[(self.concept['img'] == '') & (self.concept['modality'] == 'MM')].drop_duplicates()
        concepts['c_modality'] = concepts['concept'].map(lambda x: x.split(', ')[0])
        concepts['concept'] = concepts['concept'].map(lambda x: x.split(', ')[-1])
        concepts = concepts.sort_values(by=['name', 'score'], ascending=[True, False])
        return concepts

    @cache_df(cache=cache)
    def _get_topk(self, concepts, column='modality', k=None):
        concepts = self.get_topk(concepts, column, k)
        return concepts

    def get_single_modality_concepts(self):
        single = self._format_concepts(save_path=f'{self.save_path}/single_format.csv', name='single')
        single = self._get_topk(save_path=f'{self.save_path}/single_num_by_modality.csv', concepts=single)
        return single

    def get_single_modality_topk_concepts(self, k):
        single = self._format_concepts(save_path=f'{self.save_path}/single_format.csv', name='single')
        single = self._get_topk(save_path=f'{self.save_path}/single_top{k}.csv', concepts=single, k=k)
        return single

    def get_img_concepts(self):
        img = self._format_concepts(save_path=f'{self.save_path}/img_format.csv', name='img')
        img = self._get_topk(save_path=f'{self.save_path}/img_num_by_modality.csv', concepts=img)
        return img

    def get_img_topk_concepts(self, k):
        img = self._format_concepts(save_path=f'{self.save_path}/img_format.csv', name='img')
        img = self._get_topk(save_path=f'{self.save_path}/img_top{k}.csv', concepts=img, k=k)
        return img

    def get_mm_concepts(self):
        mm = self._format_concepts(save_path=f'{self.save_path}/mm_format.csv', name='mm')
        return mm

    def get_mm_concepts_num_by_modality(self):
        mm = self._format_concepts(save_path=f'{self.save_path}/mm_format.csv', name='mm')
        mm = self._get_topk(save_path=f'{self.save_path}/mm_num_by_modality.csv', concepts=mm, column='c_modality')
        return mm

    def get_mm_topk_concepts(self, k):
        mm = self._format_concepts(save_path=f'{self.save_path}/mm_format.csv', name='mm')
        mm = self._get_topk(save_path=f'{self.save_path}/mm_top{k}.csv', concepts=mm, k=k)
        return mm

    def get_conceptbank_concepts(self):
        import torch
        import pandas as pd
        FA = torch.cat([i['FA'] for i in self.intermediate_score if 'FA' in i.keys() and len(i.keys()) == 1], dim=0)
        ICGA = torch.cat([i['ICGA'] for i in self.intermediate_score if 'ICGA' in i.keys() and len(i.keys()) == 1],
                         dim=0)
        US = torch.cat([i['US'] for i in self.intermediate_score if 'US' in i.keys() and len(i.keys()) == 1], dim=0)
        MM = torch.cat([torch.stack([FA, ICGA]).max(dim=0)[0], US], dim=1)
        names = [i['name'] for i in self._cache_embeddings]
        pathology = [i['pathology'] for i in self._cache_embeddings]
        concepts = []
        for name, p, scores in zip(names, pathology, MM):
            for i, score in enumerate(scores):
                concepts.append({'name': name, 'pathology': p,
                                 'concept': self.args.concept_bank.concept_names[int(i)], 'score': score.item()})
        concept = pd.DataFrame(concepts)
        concept['modality'] = concept['concept'].map(lambda x: x.split(', ')[0])
        concept['c_modality'] = concept['modality']
        concept['concept'] = concept['concept'].map(lambda x: x.split(', ')[-1])
        concept['time'] = concept['concept'].map(lambda x: '无' if '期' not in x else ('晚期' if '晚' in x else '早期'))
        # path
        concept.to_csv(f'{self.save_path}/conceptbank_concepts.csv', index=False)
        return concept
