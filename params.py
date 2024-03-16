import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    ######################################### train params #########################################
    parser.add_argument("--name", default="", type=str, help="First level dir name")

    parser.add_argument("--output_dir", default="./result", type=str,
                        help="Root dir name")

    parser.add_argument("--mark", default="", type=str, help="Second level dir name")
    parser.add_argument("--modality", default='MM', type=str, help="MRI contrast(default, normal)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", "-k", type=int, default=0)
    parser.add_argument("--out_channel", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bz", type=int, default=4)
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--no_data_aug', action='store_true', default=False)
    parser.add_argument("--idx", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--ignore', action='store_true', default=False)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--plot_curve', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--plot_after_train', action='store_true', default=False)
    parser.add_argument('--cudnn_nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    ######################################### data params #########################################
    parser.add_argument('--mix_up_alpha', type=float, default=None, help='data augmentation -- mixup')
    parser.add_argument('--imbalance', '-imb', action='store_true', default=False)

    parser.add_argument('--us_crop', action='store_true', default=False)
    parser.add_argument('--valid_only', action='store_true', default=False, help='only valid set')
    parser.add_argument('--test_only', action='store_true', default=False, help='only valid set')
    parser.add_argument('--same_valid', action='store_true', default=False, help='same valid set')
    parser.add_argument('--time_shuffle', action='store_true', default=False, help='shuffle time axis')
    parser.add_argument('--modality_shuffle', action='store_true', default=False, help='shuffle modality axis')
    parser.add_argument('--add', action='store_true', default=False, help='shuffle modality axis')
    parser.add_argument('--under_sample', '-us', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='b0')

    ######################################### model params #########################################
    parser.add_argument('--prognosis', '-p', action='store_true', default=False)
    parser.add_argument('--dummy', action='store_true', default=False, help='dummy class for imbalanced dataset')
    parser.add_argument('--bidirectional', '-bi', action='store_true', default=False, help='bidirectional LSTM')
    parser.add_argument('--fusion', '-fu', type=str, default='pool', help='fusion module: pool, lstm')
    parser.add_argument('--loss', default='CrossEntropy', type=str,
                        help='loss function - can be CrossEntropy or CrossFocal')

    ######################################### cocnept params #########################################
    parser.add_argument('--concept_bank', '-cb', default=None)
    parser.add_argument('--clip_name', default='cav', type=str,
                        help='clip model name: RN50, RN101, RN50x4, RN50x16, backbone')
    parser.add_argument('--modality_mask', default=False, action='store_true', help='modality mask')
    parser.add_argument('--cbm_model', default='mm', type=str, help='mm, s, sa')
    parser.add_argument('--init_method', default='default', type=str, help='default, zero, kaiming, concept')
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--cbm_location', default='report', type=str, help='params | file | report | human')
    parser.add_argument('--svm_C', default=.1, type=float, help='.001, .1')
    parser.add_argument('--report_shot', '-rs', default=1., type=float, help='.001, .1')
    parser.add_argument('--concept_shot', '-cs', default=1., type=float, help='.001, .1')
    parser.add_argument('--pos_samples', default=50, type=int, help='50, 100')
    parser.add_argument('--neg_samples', default=0, type=int, help='neg samples of cavs')
    parser.add_argument('--cav_split', default=0.5, type=float, help='train valid split of cavs')
    parser.add_argument('--activation', '-act', default=None, type=str, help='sigmoid, softmax')
    parser.add_argument('--analysis_select_modality', '-asm', default=True, type=bool, help='')
    parser.add_argument('--analysis_top_k', '-atk', default=None, type=int, help='')
    parser.add_argument('--analysis_threshold', '-ath', default=None, type=float, help='')
    parser.add_argument('--act_on_weight', '-aow', action='store_true', default=False)
    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--weight_norm', action='store_true', default=False)

    args = parser.parse_args()
    args.dir_name = ''
    if args.dummy:
        args.out_channel += 1
    return args


################################################### Configuration ##################################################
openai_info = {
    'api_base': 'https://openai.liuy.site/v1/',
    'api_key': 'sk-lWQydh90SCAD0olBA0EfC37c76C74fF9A8Af389cAa095b16',
    'model': 'gpt-3.5-turbo',
    'prompts': [
        {"role": "user",
         "content": "你现在是一名研究脉络膜黑色素瘤，脉络膜血管瘤，脉络膜转移癌三种疾病的医学专家。"
                    "熟悉 FA、ICGA 和 多普勒超声US 这三种模态的眼底疾病图像。"
                    "给你确诊的疾病和对应几种模态图像的临床特征描述，并根据这些临床特征描述"
                    "生成病人诊断报告。请使用Markdown格式封装，并且符合标准诊断报告格式。下面是一个参考模版：\n"
                    "# 诊断报告 \n "
                    "## 病人信息 \n - 姓名：\n - 性别：\n - 年龄：\n ## 临床特征 \n - 主诉和症状：\n"
                    "## 诊断结果 \n - 影像学检查结果：\n - 其他相关检查和发现：\n ## 诊断建议 \n - 初步诊断：\n - 治疗方案：\n - 随访计划和安排：\n "
                    "## 注意事项 \n - 病人需要特别注意的事项：\n - 预防措施和健康建议：\n "
                    "## ChatGPT生成声明 \n - 本诊断报告由ChatGPT生成，仅供参考，不作为临床诊断依据。"
         },
        {"role": "assistant", "content": "好的，请你告诉我图像的特征描述和初步诊断结果以及其他信息，我会填充内容并生成标准的诊断报告。"
                                         "如果病人信息缺失我会标注[未知]，其他信息我会根据特征描述和初步诊断结果自动生成。"}
    ]
}
tencent_info = {
    'SecretId': 'AKIDsa4ITO1OXQYT8WeP2QZVCerOfiW6RYP1',
    'SecretKey': 'RgibhcQ4TlqgHdjZ8ar24sQimeGbgW7R'
}

data_info = {
    'data_path': 'data',
    'csv_path': 'CSV/data_split'
}

################################################## data #####################################################

# seed = 42
img_size = (256, 256)
class_weight = [.8, 1, .2, 1]
standardization_int = {
    'US': {
        'mean': [0.1591, 0.1578, 0.1557],
        'std': [0.0641, 0.0645, 0.0641]
    },
    'FA': {
        'mean': [0.3476, 0.3476, 0.3476],
        'std': [0.1204, 0.1204, 0.1204]
    },
    'ICGA': {
        'mean': [0.2864, 0.2864, 0.2864],
        'std': [0.1265, 0.1265, 0.1265]
    }
}

pathology_labels = {'血管瘤': 0, '转移癌': 1, '黑色素瘤': 2, 'noise': 3,
                    'Choroidal Hemangioma': 0, 'Choroidal Metastatic Carcinoma': 1, 'Choroidal Melanoma': 2, }
pathology_labels_en = {'Choroidal Hemangioma': 0, 'Choroidal Metastatic Carcinoma': 1, 'Choroidal Melanoma': 2}
pathology_labels_cn_to_en = {'血管瘤': 'Choroidal Hemangioma', '转移癌': 'Choroidal Metastatic Carcinoma',
                             '黑色素瘤': 'Choroidal Melanoma'}
id_to_labels = {0: '血管瘤', 1: '转移癌', 2: '黑色素瘤', 3: 'noise'}
prognosis_labels = {'无转移': 0, '转移': 1, '死亡': 2, 'noise': 3}

dataset_keys = ('FA', 'US', 'ICGA', 'MM')

# 数据读取过程中，模态到数据类型的映射
modality_data_keys = {
    'FA': ('FA', 'MM'),
    'US': ('US', 'MM'),
    'ICGA': ('ICGA', 'MM'),
    'MM': dataset_keys,
    'MMO': ('MM',),
    'MMs': dataset_keys,
    'MMm': dataset_keys,
}

# trainer中，数据模态到数据类型的映射
# MMO MM数据集上训练，不交替训练
# MMS ALL数据集上训练，不交替训练
# MMm MM数据集上训练，交替训练
evaluation = {
    'FA': ('FA',),
    'US': ('US',),
    'ICGA': ('ICGA',),
    'MM': dataset_keys,
    'MMO': dataset_keys,
    'MMs': dataset_keys,
    'MMm': dataset_keys
}
modality_data_map = {
    'train': {
        'FA': ('FA',),
        'US': ('US',),
        'ICGA': ('ICGA',),
        'MM': dataset_keys,
        'MMO': ('MM',),
        'MMs': ('MM',),
        'MMm': ('MM', 'MM', 'MM')},
    'valid': evaluation,
    'test': evaluation,
    'infer': evaluation
}

# 模型初始化中，模型到模态的映射
# Mapping from model to mode during model initialization.
model_modality_map = {
    'FA': ('FA',),
    'US': ('US',),
    'ICGA': ('ICGA',),
    'MM': ('FA', 'US', 'ICGA'),
}

# Logger 数据模态到模型的映射
modality_model_map = {
    'FA': ('FA',),
    'US': ('US',),
    'ICGA': ('ICGA',),
    'MM': dataset_keys,
    'MMO': dataset_keys,
    'MMs': dataset_keys,
    'MMm': dataset_keys,
}

resnet = {
    '10': {
        'block': 'ResNetBlock',
        'layers': [1, 1, 1, 1],
        'lstm': 2,
    },
    '18': {
        'block': 'ResNetBlock',
        'layers': [2, 2, 2, 2],
        'block_inplanes': [64, 128, 256, 512],
        'lstm': 2,

    },
    '34': {
        'block': 'ResNetBlock',
        'layers': [3, 4, 6, 3],
        'block_inplanes': [64, 128, 256, 512],
        'lstm': 2,

    },
    '50': {
        'block': 'ResNetBottleneck',
        'layers': [3, 4, 6, 3],
        'block_inplanes': [64, 128, 256, 512],
        'lstm': 2,
    },
    '101': {
        'block': 'ResNetBottleneck',
        'layers': [3, 4, 23, 3],
        'block_inplanes': [64, 128, 256, 512],
        'lstm': 2,

    },
}
################################################### interpretation ###################################################
concepts_conf = {
    'clip_name': 'RN50',
    # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    'dir_path': 'concepts_saved',
    'csv_path': 'CSV/concept',
}
