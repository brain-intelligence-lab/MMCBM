import os

import torch
from collections import defaultdict
import numpy as np

from sklearn.svm import SVC
from tqdm import tqdm

from params import img_size

###################################################### CLip ##############################################
from models import Clip, MedClip, BioMedClip

###################################################### Concept Loader ##############################################
import pandas as pd

from utils.logger import char_color


class ConceptsLoader:
    def __init__(self,
                 root_dir,
                 location,
                 backbone,
                 device,
                 transfomer,
                 report_shot=1.,
                 concept_shot=1.,
                 exclude_data_path=None,
                 language='zh'):
        from params import data_info
        from ast import literal_eval
        self.root_dir = root_dir
        self.location = location
        self.report_shot = report_shot
        self.concept_shot = concept_shot
        self.exclude_data_path = exclude_data_path
        self.language = language
        dfs = []

        def flatten(x):
            pathology = x['pathology']
            path = x['path']
            type = x['type']
            for modality, path in path.items():
                for idx in range(len(path)):
                    dfs.append(
                        {'path': path[idx], 'name': x['name'], 'pathology': pathology, 'type': type[modality][idx],
                         'modality': modality})
            return None

        pd.read_csv(f'{data_info["csv_path"]}/DATA_Cleaned.csv').agg(
            lambda x: x.map(literal_eval) if '{' in str(x.iloc[0]) else x)[['path', 'name', 'pathology', 'type']].apply(
            flatten, axis=1)
        self.data_df = pd.DataFrame(dfs)
        self.backbone = backbone
        self.device = device
        self.transformer = transfomer
        self.pos_neg_concepts_loader = []
        self.init_concept_map()

    def init_concept_map(self):
        file_path = f'{self.root_dir}/{self.location}_concepts_map_rshot{self.report_shot}_csho{self.concept_shot}.csv'
        if os.path.exists(file_path):
            self.concepts = pd.read_csv(file_path)
            return

        if 'report' in self.location:
            concepts = []
            o_path = f'CSV/concept/patient_concepts_map.csv'
            report = pd.read_csv(o_path)
            if self.exclude_data_path is not None:
                exclude_data = pd.read_csv(self.exclude_data_path)
                report = report.loc[~report['name'].isin(exclude_data['name'])]
            if self.report_shot != 1:
                def r_ample(x):
                    sample_size = max(1, int(len(x['name'].drop_duplicates()) * self.report_shot))
                    return x.sample(n=sample_size)

                rs = report.groupby(['pathology']).apply(r_ample).reset_index(drop=True)
                # import pdb
                # pdb.set_trace()
                report = report.loc[report['name'].isin(rs['name'])]

            for _, raw in report.iterrows():
                concept = raw['concept']
                imgs = self.data_df[
                    (self.data_df['name'] == raw['name']) &
                    (self.data_df['modality'] == raw['modality'])].to_dict('records')
                if '期' in raw['time'] and raw['modality'] != 'US':
                    if '晚' in raw['time']:
                        concepts.extend(
                            [{'concept': raw['time'] + concept, 'path': d['path'], 'modality': d['modality']} for d in
                             imgs if d['type'] in ['later']])
                    elif '中' in raw['time']:
                        pass
                    else:
                        concepts.extend(
                            [{'concept': raw['time'] + concept, 'path': d['path'], 'modality': d['modality']} for d in
                             imgs if d['type'] in ['early', 'middle']])
                else:
                    concepts.extend(
                        [{'concept': concept, 'path': img['path'], 'modality': img['modality']} for img in
                         imgs])
            self.concepts = self.translate(pd.DataFrame(concepts), o_path)
        elif 'human' in self.location and 'clean' not in self.location:
            o_path = f'CSV/concept/human_concepts_map.csv'
            human_concept = pd.read_csv(o_path).fillna('')
            self.concepts = self.translate(human_concept.loc[human_concept['report'] == True], o_path)
        elif 'human' in self.location and 'clean' in self.location:
            o_path = f'CSV/concept/human_patient_concepts_map.csv'
            human_concept = pd.read_csv(o_path).fillna('')
            human_concept['concept'] = human_concept['time'] + human_concept['concept']
            self.concepts = self.translate(human_concept.loc[human_concept['report'] == True], o_path)
        elif 'human' in self.location and 'report' in self.location:
            o_path = 'CSV/concept/human_report_concepts_map.csv'
            self.concepts = self.translate(pd.read_csv(o_path).fillna(''), o_path)
        else:
            raise ValueError(f"Unknown location {self.location}")
        self.concepts['concept'] = self.concepts['modality'] + ', ' + self.concepts['concept']
        if self.concept_shot != 1:
            cs = self.concepts.groupby('modality').apply(
                lambda x: x['concept'].drop_duplicates().sample(frac=self.concept_shot)).reset_index(drop=True)
            self.concepts = self.concepts.loc[self.concepts['concept'].isin(cs)]

        self.concepts.to_csv(file_path, index=False)

    def get_concepts(self):
        self.concepts['pathology'] = self.concepts['path'].apply(lambda x: x.split('/')[1])
        return self.concepts[['concept', 'pathology', 'modality']].drop_duplicates()

    def translate(self, df, o_path):
        if self.language == 'zh':
            return df
        trans_path = os.path.join(os.path.dirname(o_path), os.path.basename(o_path).replace('.', '_trans.'))
        if os.path.exists(trans_path):
            return pd.read_csv(trans_path)
        from utils.translator import TencentTranslator
        translator = TencentTranslator()
        translated = translator.translate(df['concept'].unique().tolist(), source='zh', target='en')
        translated = dict(zip(df['concept'].unique(), translated))
        df['concept'] = df['concept'].apply(lambda x: translated[x])
        df.to_csv(trans_path, index=False)

    def get_pos_neg_images(self, concept):
        if 'report' in self.location or 'human' in self.location:
            if 'strict' in self.location:
                pos_images = self.concepts[self.concepts['concept'] == concept][['path', 'latent']]
                neg_images = self.concepts[~self.concepts['path'].isin(pos_images['path'])][
                    ['path', 'latent']]
                pos_images = pos_images.to_dict('records')
                neg_images = neg_images.loc[neg_images['path'].drop_duplicates().index].to_dict('records')
            else:
                pos_images = self.concepts[self.concepts['concept'] == concept].to_dict('records')
                neg_images = self.concepts[self.concepts['concept'] != concept].to_dict('records')
        else:
            pos_modality = self.concepts[self.concepts['concept'] == concept].iloc[0]['modality']
            pos_images = self.data_df[(self.data_df['modality'] == pos_modality)][
                ['path', 'modality']].to_dict('records')
            neg_images = self.data_df[(self.data_df['modality'] != pos_modality)][
                ['path', 'modality']].to_dict('records')
        return pos_images, neg_images

    def init_pos_neg_concepts_loader(self, n_samples, neg_samples):
        latents_path = f'{self.root_dir}/latent_rshot{self.report_shot}_csho{self.concept_shot}.pkl'

        if os.path.exists(latents_path):
            self.latents = torch.load(latents_path, map_location='cpu')
        else:
            # 全部图像编码
            modalities = []
            paths = []
            latents = []

            for raw in tqdm(self.concepts[['modality', 'path']].drop_duplicates().to_dict("records")):
                image = raw['path']
                modality = raw['modality']
                image = self.transformer({raw['modality']: image})[raw['modality']].to(self.device).unsqueeze(
                    0).unsqueeze(0)
                latent = self.backbone.encode(modality, image).squeeze(dim=1).detach().cpu()
                modalities.append(modality)
                paths.append(raw['path'])
                latents.append(latent)
            latents = torch.cat(latents, dim=0)
            self.latents = {
                'paths': paths,
                'latents': latents
            }
            torch.save(self.latents, latents_path)

        self.latents['latents'] = list(
            map(lambda x: x.numpy(), self.latents['latents'].chunk(self.latents['latents'].shape[0])))
        self.latents = pd.DataFrame({'path': self.latents['paths'], 'latent': self.latents['latents']})
        self.concepts = self.concepts.merge(self.latents, on='path')
        for concept in self.concepts['concept'].unique():
            pos_images, neg_images = self.get_pos_neg_images(concept)
            if n_samples <= 0:
                pos_images = np.random.choice(pos_images, max(len(pos_images), len(neg_images)), replace=True)
                neg_images = np.random.choice(neg_images, max(len(pos_images), len(neg_images)), replace=True)
            else:
                try:
                    pos_images = np.random.choice(pos_images, 2 * n_samples, replace=False)
                except Exception as e:
                    print(e)
                    print(f"{len(pos_images)} positives, {len(neg_images)} negatives")
                    pos_images = np.random.choice(pos_images, 2 * n_samples, replace=True)

                try:
                    neg_images = np.random.choice(neg_images, 2 * neg_samples, replace=False)
                except Exception as e:
                    print(e)
                    print(f"{len(pos_images)} positives, {len(neg_images)} negatives")
                    neg_images = np.random.choice(neg_images, 2 * neg_samples, replace=True)

            self.pos_neg_concepts_loader.append({'concept': concept, 'pos': pos_images, 'neg': neg_images})
        torch.save(self.pos_neg_concepts_loader, f'{self.root_dir}/pos_neg_concepts_loader.pkl')

    def __call__(self, n_samples, neg_samples):
        if len(self.pos_neg_concepts_loader) == 0:
            self.init_pos_neg_concepts_loader(n_samples, neg_samples)
        return self.pos_neg_concepts_loader


class ConceptsLearner:
    def __init__(self, device, clip_name, bank_dir, location, backbone=None, report_shot=1., concept_shot=1.,
                 exclude_data_path=None, n_samples=50, neg_samples=0, svm_C=0.1, cav_split=0.5):
        # from params import concepts_conf
        # os.makedirs(concepts_conf['dir_path'], exist_ok=True)
        self.concepts = {}
        self.concept_loaders = None
        print(char_color(f"ClipName: {clip_name}", color='yellow'))
        self.clip_name = clip_name
        self.bank_dir = os.path.join(bank_dir, 'ConceptBank')
        print(char_color(f"ConceptsPath: {self.bank_dir}", color='yellow'))
        os.makedirs(self.bank_dir, exist_ok=True)

        self.location = location
        self.backbone = backbone
        self.device = device
        self.report_shot = report_shot
        self.concept_shot = concept_shot
        self.exclude_data_path = exclude_data_path
        self.download_root = 'clip_concepts_saved'
        self.n_samples = n_samples
        self.neg_samples = neg_samples if neg_samples > 0 else n_samples
        self.svm_C = svm_C
        self.cav_split = cav_split
        self.init_clip_model()

    def init_clip_model(self):
        if not hasattr(self, 'clip_model'):
            if 'cav' in self.clip_name:
                from .dataloader import val_transforms
                model, transforms = self.backbone, val_transforms(True, img_size)
                setattr(self, 'clip_model', model)
                setattr(self, 'transforms', transforms)
            elif 'med' in self.clip_name:
                setattr(self, 'clip_model', MedClip(self.device, self.clip_name))
                setattr(self, 'transforms', getattr(getattr(self, 'clip_model'), 'transforms'))
            elif 'bio' in self.clip_name:
                setattr(self, 'clip_model', BioMedClip(self.device))
                setattr(self, 'transforms', getattr(getattr(self, 'clip_model'), 'transforms'))
            else:
                setattr(self, 'clip_model', Clip(self.device, self.clip_name, self.download_root))
                setattr(self, 'transforms', getattr(getattr(self, 'clip_model'), 'transforms'))

    def init_concepts(self):
        import pandas as pd
        from params import pathology_labels
        if self.location == 'file':
            df = pd.read_csv(f'CSV/concept/gpt4_concepts.csv').to_dict('records')
            for data_i in df:
                c = f'This is a {data_i["modality"]} image, {data_i["concept"]}, {data_i["appearance"]}'
                if c not in self.concepts:
                    value = torch.zeros(3)
                else:
                    value = self.concepts[c]
                value[pathology_labels[data_i['pathology']]] = 1
                self.concepts[c] = value

        elif 'report' in self.location or 'human' in self.location:
            # self.concepts = pd.read_csv('data/reports.csv')['concept'].unique()
            self.concept_loaders = ConceptsLoader(self.bank_dir,
                                                  self.location,
                                                  backbone=self.backbone,
                                                  device=self.device,
                                                  transfomer=getattr(self, 'transforms'),
                                                  report_shot=self.report_shot,
                                                  concept_shot=self.concept_shot,
                                                  exclude_data_path=self.exclude_data_path,
                                                  language='en' if 'clip' in self.clip_name else 'zh')
            for idx, row in self.concept_loaders.get_concepts().iterrows():
                modality = {'FA': 'Fluorescein Angiography',
                            'ICGA': 'Indocyanine Green Angiography',
                            'US': 'Ultrasound'}[row['modality']]
                concept = row['concept'].split(', ')[1]
                c = f'This is a {modality} image, {concept}'
                if c not in self.concepts:
                    value = torch.zeros(3)
                else:
                    value = self.concepts[c]
                value[pathology_labels[row['pathology']]] = 1
                self.concepts[c] = value
            pass

    def get_clip_model(self):
        return getattr(self, 'clip_model')

    def get_clip_trans(self):
        return getattr(self, 'transforms')

    @torch.no_grad()
    def clip_learner(self):
        concept_dict = {}
        concept_dict_path = os.path.join(self.download_root,
                                         f"clip_concept_{self.location}_{self.clip_name.replace('/', '-')}"
                                         f"_rshot{self.report_shot}_csho{self.concept_shot}.pkl")
        if os.path.exists(concept_dict_path):
            concept_dict = torch.load(concept_dict_path, map_location=torch.device(self.device))
            print(f"Loading from {concept_dict_path}")
            print(f"{len(list(concept_dict.keys()))} concepts will be used.")
            return concept_dict
        self.init_concepts()
        for concept, cls in tqdm(self.concepts.items()):
            text_features = self.get_clip_model().encode_text(concept)
            concept_dict[concept] = (text_features, cls, self.get_clip_model().logit_scale, 0, {})
        print(f"# concepts: {len(concept_dict)}")
        torch.save(concept_dict, concept_dict_path)
        print(f"Dumped to : {concept_dict_path}")
        return concept_dict

    @torch.no_grad()
    def get_cav_embeddings(self, loader):
        from params import pathology_labels
        activations = []
        cls = torch.zeros(3)
        for data in tqdm(loader):
            activations.append(data['latent'])
            cls[pathology_labels[data['path'].split('/')[1]]] = 1
        activations = np.concatenate(activations, axis=0)
        return activations, cls

    def cav_leaner(self):
        # ToDO: Implement CAV Learner
        lib_path = os.path.join(self.bank_dir,
                                f"{self.backbone.name}_concept_{self.location}_{self.svm_C}_{self.n_samples}"
                                f"_rshot{self.report_shot}_csho{self.concept_shot}.pkl")
        if os.path.exists(lib_path):
            concept_dict = torch.load(lib_path, map_location=torch.device(self.device))
            print(f"Loading from {lib_path}")
            print(f"{len(list(concept_dict.keys()))} concepts will be used.")
            return concept_dict
        self.init_concepts()
        concept_libs = {}
        for concepts_data in self.concept_loaders(self.n_samples, self.neg_samples):
            concept_name, pos_loader, neg_loader = concepts_data['concept'], concepts_data['pos'], concepts_data['neg']
            # Get CAV for each concept using positive/negative image split
            print("Extracting Embeddings: ")
            pos_act, pos_cls = self.get_cav_embeddings(pos_loader)
            neg_act, _ = self.get_cav_embeddings(neg_loader)
            pos_split = int(pos_act.shape[0] * self.cav_split)
            neg_split = int(neg_act.shape[0] * self.cav_split)
            X_train = np.concatenate([pos_act[:pos_split], neg_act[:neg_split]], axis=0)
            X_val = np.concatenate([pos_act[pos_split:], neg_act[neg_split:]], axis=0)
            y_train = torch.concat(
                [torch.ones(pos_act[:pos_split].shape[0]), torch.zeros(neg_act[:neg_split].shape[0])],
                dim=0).numpy()
            y_val = torch.concat(
                [torch.ones(pos_act[pos_split:].shape[0]), torch.zeros(neg_act[neg_split:].shape[0])],
                dim=0).numpy()
            # else:
            #     X_train = np.concatenate([pos_act[:self.n_samples], neg_act[:self.neg_samples]], axis=0)
            #     y_train = torch.concat(
            #         [torch.ones(pos_act[:self.n_samples].shape[0]), torch.zeros(neg_act[:self.neg_samples].shape[0])],
            #         dim=0).numpy()
            #
            #     X_val = np.concatenate([pos_act[self.n_samples:], neg_act[self.neg_samples:]], axis=0)
            #     y_val = torch.concat(
            #         [torch.ones(pos_act[self.n_samples:].shape[0]), torch.zeros(neg_act[self.neg_samples:].shape[0])],
            #         dim=0).numpy()

            svm = SVC(C=self.svm_C, kernel="linear")
            svm.fit(X_train, y_train)
            train_acc = svm.score(X_train, y_train)
            test_acc = svm.score(X_val, y_val)
            train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
            margin_info = {
                "max": np.max(train_margin),
                "min": np.min(train_margin),
                "train_acc": train_acc,
                "test_acc": test_acc,
                "pos_mean": np.nanmean(train_margin[train_margin > 0]),
                "pos_std": np.nanstd(train_margin[train_margin > 0]),
                "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                "neg_std": np.nanstd(train_margin[train_margin < 0]),
                "q_90": np.quantile(train_margin, 0.9),
                "q_10": np.quantile(train_margin, 0.1),
                "pos_count": y_train.sum(),
                "neg_count": (1 - y_train).sum(),
            }
            margin_info = {k: torch.tensor(v).float() for k, v in margin_info.items()}
            # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
            concept_libs[concept_name] = (torch.from_numpy(svm.coef_).float(), pos_cls, 1,
                                          torch.from_numpy(svm.intercept_).float(), margin_info)
            print(concept_name, self.svm_C, train_acc, test_acc)

        # Save CAV results
        torch.save(concept_libs, lib_path)
        print(f"Saved to: {lib_path}")
        return concept_libs
        # total_concepts = len(concept_libs[C].keys())
        # print(f"File: {lib_path}, Total: {total_concepts}")

    def __call__(self):
        if 'clip' in self.clip_name:
            return self.clip_learner()
        elif 'cav' in self.clip_name and self.backbone is not None:
            return self.cav_leaner()
        raise KeyError(f"Unknown mode: {self.clip_name}")


##################################################### Concept Bank ################################################
class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AnyKey:
    def __init__(self, d):
        self.d = d

    def __getitem__(self, item):
        return self.d


class ConceptBank:
    def __init__(self, device, clip_name, location, backbone,
                 n_samples=50, neg_samples=0, svm_C=0.1, single_modality_score=False,
                 bank_dir='', report_shot=1., concept_shot=1.,
                 exclude_data_path=None, sort=True, save=True, cav_split=0.5):
        self.__name__ = 'ConceptBank'
        all_vectors, all_cls, concept_names, all_scales, all_intercepts = [], [], [], [], []
        all_margin_info = defaultdict(list)
        path = os.path.join(bank_dir, 'ConceptBank', f"concept_bank.pkl")
        self.clip_name = clip_name
        self.sort = sort
        self.single_modality_score = single_modality_score
        self.concept_leaner = ConceptsLearner(device,
                                              clip_name=clip_name,
                                              bank_dir=bank_dir,
                                              location=location,
                                              backbone=backbone,
                                              report_shot=report_shot,
                                              concept_shot=concept_shot,
                                              exclude_data_path=exclude_data_path,
                                              n_samples=n_samples,
                                              neg_samples=neg_samples,
                                              svm_C=svm_C,
                                              cav_split=cav_split
                                              )
        self.clip_model = self.concept_leaner.get_clip_model()
        self.transform = self.concept_leaner.get_clip_trans()
        # used for modality split. e.g. FA_ICGA: FA, ICGA
        self.modality_map = {'FA': 'FA', 'ICGA': 'ICGA', 'US': 'US', 'MM': 'FA_ICGA_US'}

        if os.path.exists(path) and save:
            print(char_color(f"Loading from {path}", color='green'))
            data = torch.load(path, map_location=torch.device(device))
            self.modality_mask = data['modality_mask']
            self.vectors = data['vectors']
            self.cls_weight = data['cls_weight']
            self.norms = data['norms']
            self.scales = data['scales']
            self.intercepts = data['intercepts']
            self.concept_names = data['concept_names']
            self.hidden_dim = data['hidden_dim']
        else:
            self.concepts_dict = self.concept_leaner()
            self.modality_mask = {
                m: torch.zeros(len(self.concepts_dict), dtype=torch.float, device=device) for m in
                ['FA', 'ICGA', 'US']
            }

            for idx, (concept, (tensor, cls, scale, intercept, margin_info)) in enumerate(
                    sorted(self.concepts_dict.items()) if self.sort else self.concepts_dict.items()
            ):
                try:
                    for modality in self.modality_mask.keys():
                        if modality in concept or {'FA': 'Fluorescein Angiography',
                                                   'ICGA': 'Indocyanine Green Angiography',
                                                   'US': 'Ultrasound'}[modality] in concept:
                            self.modality_mask[modality][idx] = 1
                except:
                    import pdb
                    pdb.set_trace()
                all_vectors.append(tensor)
                all_cls.append(cls)
                concept_names.append(concept)
                all_intercepts.append(torch.tensor(intercept).reshape(1, 1))
                all_scales.append(torch.tensor(scale).reshape(1, 1))
                for key, value in margin_info.items():
                    if key != "train_margins":
                        all_margin_info[key].append(value.reshape(1, 1))

            for key, val_list in all_margin_info.items():
                margin_tensor = torch.tensor(
                    torch.concat(val_list, dim=0), requires_grad=False
                ).float().to(device)
                all_margin_info[key] = margin_tensor

            self.vectors = torch.concat(all_vectors, dim=0).float().to(device)
            self.cls_weight = torch.stack(all_cls, dim=1).float().to(device)
            self.norms = torch.norm(self.vectors, p=2, dim=1, keepdim=True).detach()
            self.scales = torch.concat(all_scales, dim=0).float().to(device)
            self.intercepts = torch.concat(all_intercepts, dim=0).float().to(device)
            self.concept_names = concept_names
            self.hidden_dim = self.vectors.shape[1]
            print("Concept Bank is initialized.")
            torch.save({
                'modality_mask': self.modality_mask,
                'vectors': self.vectors,
                'cls_weight': self.cls_weight,
                'norms': self.norms,
                'scales': self.scales,
                'intercepts': self.intercepts,
                'concept_names': self.concept_names,
                'hidden_dim': self.hidden_dim,
            }, path)

    def set_modality_map(self, modality_map):
        self.modality_map.update(modality_map)

    def set_single_modality_score(self, single_modality_score):
        self.single_modality_score = single_modality_score

    @property
    def n_concepts(self):
        if 'MM' not in self.modality:
            n = 0
            for m in self.modality_mask.keys():
                if m in self.modality:
                    n += len(self.modality_mask[m].flatten().nonzero().flatten())
            return n
        return self.vectors.shape[0]

    def __getitem__(self, modality):
        self.modality = self.modality_map[modality] if modality in self.modality_map.keys() else modality
        return self

    def modality_score_reshape(self, score):
        if score.ndim == 1:
            score = score.unsqueeze(dim=0)
        mask = 0
        for m in self.modality_mask.keys():
            if m in self.modality:
                mask = mask + self.modality_mask[m].flatten()
        if score.shape[-1] == self.vectors.shape[0]:
            indexes = mask.nonzero().flatten()
            return score[:, indexes]
        else:
            new_score = torch.ones((score.shape[0], self.vectors.shape[0]), device=score.device) * -10
            indexes = mask.nonzero().flatten()
            new_score[:, indexes] = score
            return new_score

    def compute_dist(self, emb, m_mask=False):
        if 'clip' in self.clip_name:
            margins = (emb @ self.vectors.T) * self.scales.T
        elif 'cav' in self.clip_name:
            if 'cav1' in self.clip_name:
                margins = (self.scales * (self.vectors @ emb.T) + self.intercepts).T
            elif 'cav2' in self.clip_name:
                vectors = self.vectors / self.norms
                emb = emb / torch.norm(emb, p=2, dim=1, keepdim=True).detach()
                margins = (self.scales * (vectors @ emb.T) + self.intercepts).T
            else:
                # Computing the geometric margin to the decision boundary specified by CAV.
                # margins = (self.scales * (torch.matmul(self.vectors, emb.T) + self.intercepts) / (self.norms)).T
                margins = ((self.scales * (self.vectors @ emb.T) + self.intercepts) / self.norms).T
        else:
            raise KeyError(f"Unknown mode: {self.clip_name}")

        if 'norm' in self.clip_name:
            margins = margins / torch.norm(margins, p=2, dim=1, keepdim=True).detach()
        elif 'softmax' in self.clip_name:
            margins = margins.softmax(dim=-1)
        elif 'tanh' in self.clip_name:
            margins = torch.tanh(margins)

        if self.single_modality_score:
            return self.modality_score_reshape(margins)

        if m_mask:
            mask = self.modality_mask[self.modality].repeat(margins.shape[0], 1)
            return margins * mask + (1 - mask) * margins.min(dim=1, keepdim=True)[0]

        return margins

    def get_concept_from_threshold(self, attn_score: torch.Tensor, threshold, modality=None):
        scores, concepts = [], []
        if self.single_modality_score:
            attn_score = self.modality_score_reshape(attn_score).flatten()
        if isinstance(attn_score, torch.Tensor):
            attn_score = attn_score.cpu().numpy()
        for idx, (score, concept) in enumerate(zip(attn_score, self.concept_names)):
            if modality is not None and modality not in concept and modality != 'MM':
                continue
            if threshold is None or score > threshold:
                scores.append(score)
                concepts.append(concept)
        return scores, concepts

    def get_topk_concepts(self, attn_score, k=5, sign=1, modality=None):
        if modality is not None and modality != 'MM':
            if self.single_modality_score:
                attn_score = self.modality_score_reshape(attn_score)
            for indice, concept in enumerate(self.concept_names):
                if modality not in concept:
                    attn_score[indice] = 0

        topk_scores, topk_indices = torch.topk(attn_score, k=k)
        topk_scores = topk_scores.cpu().numpy()
        topk_scores = topk_scores * sign
        topk_indices = topk_indices.detach().cpu().numpy()
        scores, concepts = [], []
        for j in topk_indices:
            concepts.append(self.concept_names[j % len(self.concept_names)])
        return topk_scores, concepts
