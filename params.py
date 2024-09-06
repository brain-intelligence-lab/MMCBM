import argparse


def str_or_bool(value):
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    return value


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


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

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bz", type=int, default=4)
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--no_data_aug', action='store_true', default=False)
    parser.add_argument("--idx", default=None, type=int)
    parser.add_argument("--bidx", default=120, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--ignore', action='store_true', default=False)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--cache_data', action='store_true', default=False)
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
    parser.add_argument('--extra_data', type=str_or_bool, default=False, help='extra test set')
    parser.add_argument('--same_valid', action='store_true', default=False, help='same valid set')
    parser.add_argument('--time_shuffle', action='store_true', default=False, help='shuffle time axis')
    parser.add_argument('--modality_shuffle', action='store_true', default=False, help='shuffle modality axis')
    parser.add_argument('--under_sample', '-us', action='store_true', default=False)
    parser.add_argument('--n_shot', type=int_or_str, default='full')
    parser.add_argument('--model', type=str, default='b0')

    ######################################### model params #########################################
    parser.add_argument('--clip_nonorm', action='store_true', default=False)
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
    parser.add_argument('--occ_act', default='abs', type=str, help='sigmoid, softmax')

    args = parser.parse_args()
    args.dir_name = ''
    if args.dummy:
        args.out_channel += 1
    return args


################################################### Configuration ##################################################
openai_info = {
    'api_base': '',
    'api_key': '',
    'model': 'gpt-3.5-turbo',
    'prompts': [
        {"role": "user",
         "content": "You are now a medical expert specializing in the study of three diseases: choroidal melanoma, choroidal hemangioma, and choroidal metastatic carcinoma."
                    "Familiar with the three modalities of fundus disease images: FA, ICGA, and Doppler ultrasound US."
                    "Provide a clinical description of the diagnosed disease and its corresponding clinical features in several modalities, based on these clinical feature descriptions."
                    "Generate patient diagnosis report. Please use Markdown format and adhere to standard diagnostic report format. Below is a reference template:\n"
                    "# Diagnosis Report \n "
                    "## Patient information \n - name：\n - gender：\n - age：\n ## Clinical features \n - Chief complaint and symptoms：\n"
                    "## Diagnostic results \n - Imaging examination results：\n - Other relevant examinations and findings：\n ## Diagnostic recommendations \n - Preliminary diagnosis：\n - Treatment plan：\n - Follow-up plan and arrangement：\n "
                    "## Precautions \n - Precautions for patients to pay special attention to：\n - Preventive measures and health advice：\n "
                    "## ChatGPT generates statements. \n - This diagnostic report is generated by ChatGPT, for reference only and should not be used as a basis for clinical diagnosis."
         },
        {"role": "assistant",
         "content": "Okay, please tell me the feature description and preliminary diagnosis results of the image, as well as any other information. I will fill in the content and generate a standard diagnostic report."
                    "If patient information is missing, I will mark it as [unknown]. Other information will be generated based on the description of characteristics and preliminary diagnosis results."}
    ]
}
tencent_info = {
    'SecretId': '',
    'SecretKey': ''
}

data_info = {
    'data_path': 'data',
    'csv_path': 'CSV/data_split'
}

################################################## data #####################################################

# seed = 42
img_size = (256, 256)
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

# modalities = ('FA', 'ICGA')
modalities = ('FA', 'ICGA', 'US')
dataset_keys = (*modalities, 'MM')

# data loader - modalities to data types
modality_data_keys = {
    'FA': ('FA', 'MM'),
    'US': ('US', 'MM'),
    'ICGA': ('ICGA', 'MM'),
    'MM': dataset_keys,
    'MMOnly': ('MM',),
}

# trainer - data modalities to data types
modality_data = {
    'FA': ('FA',),
    'US': ('US',),
    'ICGA': ('ICGA',),
    'MM': dataset_keys,
    'MMOnly': dataset_keys,
}

modality_data_map = {
    'train': modality_data,
    'valid': modality_data,
    'test': modality_data,
    'infer': modality_data
}

# Logger - Mapping from data modality to model modality
modality_model_map = {
    'FA': ('FA',),
    'US': ('US',),
    'ICGA': ('ICGA',),
    'MM': dataset_keys,
    'MMOnly': dataset_keys,
}
