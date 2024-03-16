import os.path
import random

import pandas as pd
from ast import literal_eval
from utils.logger import char_color


class DataSplit:
    def __init__(self, data_path="../data", csv_path='./CSV', prognosis=False,
                 valid_only=False, test_only=False, same_valid=False, angle=None,
                 under_sample=False, exist_ok=False, save_csv=True,
                 modality_shuffle=False, print_info=True,
                 um_3m_data=False):
        self.valid_only = valid_only
        self.test_only = test_only
        self.angle = angle
        self.same_valid = same_valid
        self.prognosis = prognosis
        self.under_sample = under_sample
        self.exist_ok = exist_ok
        self.save_csv = save_csv
        self.modality_shuffle = modality_shuffle
        self.um_3m_data = um_3m_data  # UM_all: only use 3 modality data for UM
        self.print_info = print_info

        self.data_path = data_path
        self.csv_path = csv_path
        os.makedirs(csv_path, exist_ok=True)
        self.modality = ['FA', 'ICGA', 'US']
        self.us_type_list = ['doppler_notag', 'gray_notag', 'gray_tag', 'tag', 'e', 'm', 'l']

        self.test = self._read_df('test')
        self.train = self._read_df('train')
        self.k_fold = self._read_df('k_fold')
        self.key = {
            'FA': ['FA', 'FA&US', 'FA&ICGA'],
            'ICGA': ['ICGA', 'ICGA&US', 'FA&ICGA'],
            'US': ['US', 'FA&US', 'ICGA&US'],
            'MM': ['FA&ICGA&US'],
        }

    @property
    def df(self):
        if hasattr(self, '_df'):
            return self._df
        self._df = self._get_df()
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def _file_name(self, name):
        if self.prognosis and name is not None:
            name = f'{name}_prognosis'
        if self.valid_only and name is not None:
            name = f'{name}_valid_only'
        if self.same_valid and name is not None:
            name = f'{name}_same_valid'
        if self.under_sample and name is not None:
            name = f'{name}_under_sample'
        if self.modality_shuffle and name is not None:
            name = f'{name}_modality_shuffle'
        if self.um_3m_data and name is not None:
            name = f'{name}_um_3m_data'
        return name

    def _under_sample(self):
        if self.prognosis:
            by = 'result'
        else:
            by = 'pathology'

        def _sample(x):
            num = x.groupby(by).count().min()[0]
            x = x.groupby(by).apply(lambda x: x.sample(n=num)).reset_index(drop=True)
            return x

        self.df = self.df.groupby('modality').apply(_sample).reset_index(drop=True)

    def _select_prognosis(self):
        prognosis = pd.read_csv(os.path.join(self.csv_path, 'Prognosis.csv'))[['姓名', '结局']].rename(
            columns={'姓名': 'name', '结局': 'result'})
        self.df = pd.merge(self.df, prognosis, on='name', how='inner')

    def _get_df(self):
        df = self._read_df(path=f'{self.csv_path}/DATA_Cleaned.csv')
        if self.um_3m_data:
            df = df[~((df['pathology'] == '黑色素瘤') & (df['modality'].map(lambda x: len(x.split('&'))) != 3))]

        if df is not None:
            return df
        # 生成所有数据的详细信息
        df = []
        for pathology in os.listdir(os.path.join(self.data_path)):
            if 'DS_Store' in pathology:
                continue
            for patient in os.listdir(os.path.join(self.data_path, pathology)):
                if 'DS_Store' in patient:
                    continue
                for modality in os.listdir(os.path.join(self.data_path, pathology, patient)):
                    if 'DS_Store' in modality:
                        continue
                    imgs = [os.path.join(self.data_path, pathology, patient, modality, img) for img in
                            os.listdir(os.path.join(self.data_path, pathology, patient, modality))
                            if 'png' in img or 'jpg' in img]
                    if len(imgs) != 3 and modality != 'US':
                        raise
                    imgs = sorted(imgs, key=lambda x: x.split('_')[-1])
                    df.append({
                        'path': imgs,
                        'pathology': pathology,
                        'modality': modality,
                        'name': patient,
                        'time': [img.split('_')[0] if modality == 'US' else img.split('_')[-3] for img in imgs],
                        'pos': ['None' if modality == 'US' else img.split('_')[1] for img in imgs],
                        'angle': ['None' if modality == 'US' else img.split('_')[-2] for img in imgs],
                        'type': ['None'] if modality == 'US' else ['early', 'middle', 'later']
                    })

        df = pd.DataFrame(df)

        # 成组， 列表形式保存
        def group(x):
            line = x.iloc[0].copy()
            line['modality'] = '&'.join(x['modality'].values)
            line['path'] = {k: v for k, v in zip(x['modality'].values, x['path'].values)}
            line['time'] = {k: v for k, v in zip(x['modality'].values, x['time'].values)}
            line['pos'] = {k: v for k, v in zip(x['modality'].values, x['pos'].values)}
            line['type'] = {k: v for k, v in zip(x['modality'].values, x['type'].values)}
            line['angle'] = {k: v for k, v in zip(x['modality'].values, x['angle'].values)}
            return line

        df = df.groupby(['pathology', 'name']).apply(group).reset_index(drop=True)[
            ['name', 'path', 'pathology', 'modality', 'type', 'time', 'pos', 'angle']]
        df.to_csv(f'{self.csv_path}/DATA_Cleaned.csv', index=False)
        return df

    def _read_df(self, name=None, path=None):
        name = self._file_name(name)
        if path is None:
            path = f'{self.csv_path}/{name}.csv'
        if os.path.exists(path):
            if self.print_info:
                print(char_color(f'load {name} from {path}', color='green'))
            df = pd.read_csv(path).agg(
                lambda x: x.map(literal_eval) if '{' in str(x.iloc[0]) else x)
            return df
        print(char_color(f'No {name} file: {path}', color='red'))
        return None

    def _save_df(self, name):
        df = eval(f'self.{name}')
        name = self._file_name(name)
        if self.save_csv:
            df.to_csv(f'{self.csv_path}/{name}.csv', index=False)

    def _select(self, modality, df=None):
        def is_modality(modality_list, modality2):
            # key 对应的键值，只要有一个键值在 modality2 中，就返回 True
            result = []
            modality2 = modality2.split('&')
            for modality1 in modality_list:
                modality1 = modality1.split('&')
                if len(modality1) != len(modality2):
                    result.append(False)
                else:
                    result.append(all([m in modality2 for m in modality1]))
            return any(result)

        if df is None:
            df = self.df
        df = df[df['modality'].map(lambda x: is_modality(self.key[modality], x))].copy()
        if modality != 'MM':
            # 保证单一模态
            df = df.agg(lambda x: x.map(lambda y: {modality: y[modality]}) if isinstance(x.iloc[0], dict) else x)
        df['modality'] = modality
        return df

    def get_train_test_data(self):
        if self.test is not None and self.train is None:
            self.train = self.df[~(self.df['name'].isin(self.test['name']) & self.df['pathology'].isin(
                self.test['pathology']))]
        elif self.train is not None and self.test is None:
            self.test = self.df[~(self.df['name'].isin(self.train['name']) & self.df['pathology'].isin(
                self.train['pathology']))]
        elif self.train is None and self.test is None:
            self.test = self._select('MM').groupby(['pathology']).apply(
                lambda x: x.sample(frac=.2, replace=False, axis=0)).reset_index(level=[0], drop=True)
            self.train = self.df[~self.df.index.isin(self.test.index)]

        self._save_df('test')
        self._save_df('train')

    def k_fold_split(self, modality, k):
        def fn(x):
            if not self.same_valid or modality == 'MM':
                fold_id = [i for i in range(k) for j in range(0, (len(x) - 1) // k + 1) if j * k + i < len(x)]
                random.shuffle(fold_id)
            else:
                fold_id = -1
            x['k'] = fold_id
            return x

        if self.prognosis:
            by = 'result'
        else:
            by = 'pathology'
        data_frame = self._select(modality, self.train).groupby(by).apply(fn)
        return data_frame

    def get_5fold_data(self, k=5):
        if self.k_fold is not None and not self.exist_ok:
            return self.k_fold

        if self.under_sample:
            self._under_sample()

        if self.prognosis:
            self._select_prognosis()

        if not self.valid_only:
            self.get_train_test_data()
        else:
            self.train = self.df
        us = self.k_fold_split('US', k)
        fa = self.k_fold_split('FA', k)
        icga = self.k_fold_split('ICGA', k)
        mm = self.k_fold_split('MM', k)
        self.k_fold = pd.concat([us, fa, icga, mm], axis=0).reset_index(drop=True)
        self._save_df('k_fold')

    def get_data_split(self, modality, k):
        self.get_5fold_data()
        if self.test_only:
            train, val = self.k_fold.drop(columns='k').reset_index(drop=True), None
            train = train[train['modality'] == modality]
        else:
            if self.valid_only:
                train = self.k_fold[(self.k_fold['k'] != k) | (self.k_fold['k'] == -1)].drop(columns='k').reset_index(
                    drop=True)
                val = self.k_fold[(self.k_fold['k'] == k)].drop(columns='k').reset_index(
                    drop=True)
            else:
                val = self.k_fold[self.k_fold['k'] == k].drop(columns='k').reset_index(drop=True)
                train = self.k_fold[self.k_fold['k'] != k].drop(columns='k').reset_index(drop=True)
            train = train[train['modality'] == modality]
            val = val[val['modality'] == modality]
        if self.modality_shuffle:
            def m_shuffle(x):
                path = pd.DataFrame(x['path'].to_list())
                path['FA'] = path['FA'].sample(frac=1, replace=False, axis=0).reset_index(drop=True)
                path['ICGA'] = path['ICGA'].sample(frac=1, replace=False, axis=0).reset_index(drop=True)
                path['US'] = path['US'].sample(frac=1, replace=False, axis=0).reset_index(drop=True)
                x['name'] = x['name'] + '_shuffle'
                x['path'] = path.to_dict(orient='records')
                return x

            train_mm = train[train['modality'].map(lambda x: len(x.split('&'))) == 3]
            train_mm = train_mm.groupby(['pathology']).apply(m_shuffle).reset_index(drop=True)
            train = pd.concat([train, train_mm], axis=0)
        return train, val

    def get_test_data(self):
        self.get_5fold_data()
        return self.test


if __name__ == '__main__':
    random.seed(42)
    reader = DataSplit(data_path='../data', csv_path='../CSV/data_split', prognosis=False,
                       valid_only=False, same_valid=False, under_sample=False, exist_ok=False)
    train, val = reader.get_data_split('MM', 0)
    pass
