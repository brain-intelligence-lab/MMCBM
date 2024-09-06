# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os.path
import random

import pandas as pd
from ast import literal_eval
from utils.logger import PrintColor


class DataSplit:
    def __init__(self, data_path="../data", csv_path='./CSV',
                 valid_only=False, test_only=False, same_valid=False, angle=None,
                 under_sample=False, exist_ok=False, save_csv=True,
                 modality_shuffle=False, print_info=True, extra_data=None,
                 modality=('FA', 'ICGA', 'US')):
        self.valid_only = valid_only
        self.test_only = test_only
        self.angle = angle
        self.same_valid = same_valid
        self.under_sample = under_sample
        self.exist_ok = exist_ok
        self.save_csv = save_csv
        self.modality_shuffle = modality_shuffle
        self.print_info = print_info
        self.extra_data = extra_data

        self.data_path = data_path
        self.csv_path = csv_path
        os.makedirs(csv_path, exist_ok=True)
        self.modality = modality
        self.us_type_list = ['doppler_notag', 'gray_notag', 'gray_tag', 'tag', 'e', 'm', 'l']

        if self.extra_data is not None and self.extra_data is not False:
            PrintColor(f'Using extra data for test, using mode: {self.extra_data}', color='red')
            self.test = self._read_df('test_extra', autoname=False)
        else:
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
        self._df = self._read_df(path=f'{self.csv_path}/DATA_Cleaned.csv')
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def _file_name(self, name):
        if self.valid_only and name is not None:
            name = f'{name}_valid_only'
        if self.same_valid and name is not None:
            name = f'{name}_same_valid'
        if self.under_sample and name is not None:
            name = f'{name}_under_sample'
        if self.modality_shuffle and name is not None:
            name = f'{name}_modality_shuffle'
        return name

    def _under_sample(self):
        by = 'pathology'

        def _sample(x):
            num = x.groupby(by).count().min()[0]
            x = x.groupby(by).apply(lambda x: x.sample(n=num)).reset_index(drop=True)
            return x

        self.df = self.df.groupby('modality').apply(_sample).reset_index(drop=True)

    def _read_df(self, name=None, path=None, autoname=True):
        if autoname:
            name = self._file_name(name)
        if path is None:
            path = f'{self.csv_path}/{name}.csv'
        if os.path.exists(path):
            if self.print_info:
                PrintColor(f'load {name} from {path}', color='green')
            df = pd.read_csv(path).agg(
                lambda x: x.map(literal_eval) if '{' in str(x.iloc[0]) else x)

            def del_modality(x):
                dk = []
                for key in x.keys():
                    if key not in self.modality:
                        dk.append(key)
                for key in dk:
                    x.pop(key)
                return x

            return df.agg(lambda x: x.map(lambda y: del_modality(y) if isinstance(y, dict) else y))
        PrintColor(f'No {name} file: {path}', color='red')
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

        data_frame = self._select(modality, self.train).groupby('pathology').apply(fn)
        return data_frame

    def get_5fold_data(self, k=5):
        if self.k_fold is not None and not self.exist_ok:
            return self.k_fold

        if self.under_sample:
            self._under_sample()

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
        train = self.k_fold[self.k_fold['k'] != k].drop(columns='k').reset_index(drop=True)
        val = self.k_fold[self.k_fold['k'] == k].drop(columns='k').reset_index(drop=True)
        if self.extra_data is not None and self.extra_data is not False:
            if self.extra_data is True or self.extra_data == 'True':
                inner_data = [train, val]
            elif self.extra_data == '5fold':
                inner_data = [train]
            elif self.extra_data == 'all':
                inner_data = [train, val, self._read_df('test')]
            else:
                raise ValueError(f'extra_data: {self.extra_data} is not supported')
            train = pd.concat(inner_data, axis=0).reset_index(drop=True)
            val = self._read_df('test_out', autoname=False)
        if self.test_only:
            PrintColor("test only")
            train = pd.concat([train, val], axis=0).reset_index(drop=True)
            val = None

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
