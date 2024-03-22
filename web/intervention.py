# -*- encoding: utf-8 -*-
"""
@File    :   train_GIRNet_MIxInpMSE.py
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/6 9:12 PM   liu.yang      1.0         None
"""
import os
import gradio as gr

from params import pathology_labels_cn_to_en, data_info
from utils.dataloader import *
from inference import Infer


class Intervention:
    def __init__(self,
                 json_path,
                 device='cpu',
                 normalize=True,
                 **kwargs, ):
        self.bottomk_sliders = []
        self.topk_sliders = []
        self.predictor = Infer(
            json_path=json_path,
            device=device,
            language='en',
            normalize=normalize,
            **kwargs
        )
        self.file_path = self.predictor.save_path

    def get_test_data(self, num_of_each_pathology=None, names=None, mask=True):
        test = DataSplit(data_path=data_info['data_path'], csv_path=data_info['csv_path']).get_test_data()
        if num_of_each_pathology is not None and names is None:
            test = test.groupby('pathology').head(num_of_each_pathology)
        elif names is not None:
            test = test[test['name'].isin(names)]

        def _fn(x):
            if mask:
                path = [int(random.random() * 10000000000), pathology_labels_cn_to_en[x['pathology']]]
            else:
                path = [x['name'], x['pathology']]
            [path.extend(x['path'][m]) for m in ['FA', 'ICGA', 'US']]
            return path

        test = test.apply(_fn, axis=1).to_list()
        return test

    def set_topk_sliders(self, sliders):
        self.topk_sliders = sliders

    def set_bottomk_sliders(self, sliders):
        self.bottomk_sliders = sliders

    def get_attention_matrix(self):
        return self.predictor.get_attention_matrix()

    def set_attention_matrix(self, attention_matrix):
        self.predictor.set_attention_matrix(attention_matrix)
        self.attention_score = self.predictor.get_attention_score()

    def predict_concept(self, name, pathology, language, imgs):
        inp = dict(FA=imgs[:3], ICGA=imgs[3:6], US=imgs[6:])
        self.attention_score = self.predictor.get_attention_score(inp=inp, name=name, pathology=pathology)
        self.top_k_concepts, self.top_k_values, self.indices = self.predictor.predict_topk_concepts(
            self.attention_score,
            top_k=0,  # all concepts
            language=language
        )

    def predict_topk_concept(self, *args):
        name, pathology = args[:2]
        imgs = args[2:-2]
        top_k, language = args[-2:]

        self.predict_concept(name, pathology, language, imgs)
        sliders = []
        for i in range(len(self.topk_sliders)):
            if i < top_k:
                c, s = self.top_k_concepts[i], self.top_k_values[i]
                sliders.append(
                    gr.Slider(minimum=round(self.attention_score.min().item(), 1),
                              maximum=round(self.attention_score.max().item(), 1), step=0.01,
                              label=f'{i + 1}-{c}', value=s, visible=True)
                )
            else:
                sliders.append(gr.Slider(minimum=0, maximum=1, step=0.01, label=None, visible=False))
        return sliders

    def predict_bottomk_concept(self, bottom_k):
        # self.predict_concept(*imgs)
        sliders = []
        for i in range(1, len(self.bottomk_sliders) + 1):
            if i < bottom_k + 1:
                c, s = self.top_k_concepts[-i], self.top_k_values[-i]
                sliders.append(
                    gr.Slider(minimum=round(self.attention_score.min().item(), 1),
                              maximum=round(self.attention_score.max().item(), 1), step=0.01,
                              label=f'{i}-{c}', value=s, visible=True)
                )
            else:
                sliders.append(gr.Slider(minimum=0, maximum=1, step=0.01, label=None, visible=False))
        return sliders

    def predict_label(self, language='en'):
        labels = self.predictor.get_labels_prop(language=language)
        self.predicted = max(labels, key=lambda k: labels[k])
        return labels

    def modify(self, *args):
        top_k, bottom_k, language = args[-3:]
        result = args[:-3]
        top_k_result = result[:len(self.topk_sliders)][:top_k]
        bottom_k_result = result[len(self.topk_sliders):][:bottom_k][::-1]
        result = top_k_result + bottom_k_result
        self.attention_score = self.predictor.modify_attention_score(self.attention_score,
                                                                     [
                                                                         *self.indices[:top_k],
                                                                         *self.indices[-bottom_k:]
                                                                     ],
                                                                     result,
                                                                     inplace=True)
        labels = self.predictor.get_labels_prop(self.attention_score, language=language)
        self.predicted = max(labels, key=lambda k: labels[k])
        return labels

    def download(self, file_name):
        def _fn(language):
            concepts = os.path.join(self.file_path, file_name)
            self.predictor.concepts_to_dataframe(self.attention_score, language=language).to_csv(concepts, index=False)
            return concepts

        return _fn

    def fresh_barplot(self, language):
        df = self.predictor.concepts_to_dataframe(self.attention_score, language=language)
        return gr.BarPlot(
            df,
            x="Concept",
            y="Concept Activation Score",
            title="Concept Activation Score",
            show_label=False,
            height=150,
            width=1500
        )

    def report(self, chat_history, top_k, language):
        if hasattr(self, 'top_k_concepts'):
            top_k_concepts, top_k_values, indices = self.predictor.predict_topk_concepts(
                self.attention_score,
                top_k=top_k,  # all concepts
                language=language
            )
            yield from self.predictor.generate_report(chat_history,
                                                      top_k_concepts,
                                                      top_k_values,
                                                      self.predicted,
                                                      language=language)
        else:
            raise gr.Error("Please upload images and click 'Predict' button first!")

    def clear(self):
        if hasattr(self, 'top_k_concepts'):
            del self.top_k_concepts
        if hasattr(self, 'attention_score'):
            del self.attention_score
        if hasattr(self, 'top_k_values'):
            del self.top_k_values
        if hasattr(self, 'indices'):
            del self.indices
        if hasattr(self, 'attention_matrix'):
            del self.attention_matrix
