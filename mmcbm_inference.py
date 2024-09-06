# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

from inference import Infer
from params import pathology_labels_cn_to_en


def predict_concept(imgs, top_k=10, language='en'):
    imgs = imgs[2:]
    inp = dict(FA=imgs[:3], ICGA=imgs[3:6], US=imgs[6:])
    attention_score = predictor.get_attention_score(inp=inp)
    top_k_concepts, top_k_values, indices = predictor.predict_topk_concepts(
        attention_score,
        top_k,
        language=language
    )
    labels = predictor.get_labels_prop(attention_score, language=language)
    return top_k_concepts, top_k_values, indices, labels


if __name__ == "__main__":
    json_path = ('result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_aow_zero_MM_max/'
                 'fold_0_report_strict_r1.0_c1.0')
    backbone='Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0'
    device = 'cpu'
    normalize = True
    fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us = [
        'path/to/fa_e.jpg',
        'path/to/fa_m.jpg',
        'path/to/fa_l.jpg',
        'path/to/icga_e.jpg',
        'path/to/icga_m.jpg',
        'path/to/icga_l.jpg',
        'path/to/us.jpg',
    ]
    predictor = Infer(
        json_path=json_path,
        backbone=backbone,
        device=device,
        labels=list(pathology_labels_cn_to_en.keys()),
        labels_en=list(pathology_labels_cn_to_en.values()),
        normalize=normalize,
        idx=180,
    )
    top_k_concepts, top_k_values, indices, labels = predict_concept(
        [fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
        top_k=10,
    )
    print(top_k_concepts, top_k_values, labels)
