import os.path
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix


def plot_curve(path, csv_logger, dpi=200, hue='modality', split='stage_name'):
    path = os.path.join(path, 'CURVE', split)
    os.makedirs(path, exist_ok=True)
    sns.set_theme(style='whitegrid')
    dataframe = csv_logger.dataframe()
    metric_columns = dataframe.columns.drop(['epoch', 'stage_name', 'modality'])

    for stage in dataframe[split].drop_duplicates():
        dataframe_s = dataframe[dataframe[split] == stage]
        # if stage == 'test':
        #     plt.figure(figsize=(12, 6), dpi=dpi)
        #     sns.barplot(data=dataframe_s.melt(id_vars=['epoch', 'modality', 'stage_name']), x='variable', y='value',
        #                 hue=hue)
        #     plt.xlabel('Metrics', fontsize=15)
        #     plt.xticks(rotation=15)
        #     plt.savefig(f'{path}/{stage}.png')

        for metric in metric_columns:
            plt.figure(figsize=(10, 7), dpi=dpi)
            sns.lineplot(data=dataframe_s[~dataframe_s[metric].isna()], x='epoch', y=metric, hue=hue)
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel(f'{metric} Score', fontsize=15)
            plt.title(f'{metric} Score Plot', fontsize=15)
            plt.savefig(f'{path}/{stage}_{metric}.png')

        if split == 'modality':
            types = 'stage_name'
        else:
            types = 'modality'
        melt = dataframe_s.melt(id_vars=['epoch', 'modality', 'stage_name'], var_name='metric', value_name='score')
        melt = melt[melt['metric'] != 'loss']
        for _type in dataframe_s[types].drop_duplicates():
            for _tt in ['macro', 'weight']:
                _melt = melt[melt.apply(lambda x: _tt in x['metric'] or 'accuracy' in x['metric'], axis=1)]
                plt.figure(figsize=(10, 7), dpi=dpi)
                sns.lineplot(data=_melt[_melt[types] == _type], x='epoch', y='score', hue='metric')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel(f'{_type} Score', fontsize=15)
                plt.title(f'{_type} Score Plot', fontsize=15)
                plt.savefig(f'{path}/{stage}_{_type}_{_tt}.png')

        plt.close('all')


def plot_confusion_matrix(path, stage_name, prognosis=False, csv_logger=None, min_epoch=150):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    FONT_FAMILY = 'sans-serif'  # Arial or Helvetica
    AXIS_LABEL_SIZE = 14
    TICK_LABEL_SIZE = 12

    DPI = 300
    # Apply font settings
    plt.rcParams['font.family'] = FONT_FAMILY
    if csv_logger is None:
        from utils.logger import CSVLogs
        csv_logger = CSVLogs(path, file_name='pred_output')

    if prognosis:
        display_labels = np.array(['NotMetas', 'Metas', 'Died'])
    else:
        display_labels = np.array(['Angioma', 'Metastatic', 'Melanoma'])
    path = os.path.join(path, 'Confusion_Matrix')

    os.makedirs(path, exist_ok=True)
    df = csv_logger.dataframe()
    df = df[df['stage_name'] == stage_name][['epoch', 'modality', 'labels', 'scores']]
    # select epoch > 100

    df = df[df['epoch'] > min_epoch]
    for _, raw in df.iterrows():
        preds = np.array(raw['scores'])[:, :3].argmax(-1)
        labels = np.array(raw['labels'])
        FIGURE_SIZE = (5, 4)
        cm = confusion_matrix(y_true=labels, y_pred=preds, normalize='all')

        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels[np.unique(labels)])
        # fig, ax = plt.subplots(dpi=200)
        # plt.grid(False)
        # disp.plot(cmap=plt.cm.Blues, ax=ax)

        with sns.plotting_context(rc={"font.size": TICK_LABEL_SIZE,
                                      "axes.labelsize": AXIS_LABEL_SIZE,
                                      "xtick.labelsize": TICK_LABEL_SIZE,
                                      "ytick.labelsize": TICK_LABEL_SIZE}):
            plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
            ax = sns.heatmap(cm, annot=True, fmt='.2%', cmap="Blues", xticklabels=display_labels[np.unique(labels)],
                             yticklabels=display_labels[np.unique(labels)], square=True)

            # Set colorbar labels format to percentage
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

            ax.set_xlabel('Predicted Label', fontsize=AXIS_LABEL_SIZE, labelpad=10)
            ax.set_ylabel('True Label', fontsize=AXIS_LABEL_SIZE, labelpad=10)
            # plt.title('Confusion Matrix', fontsize=TITLE_SIZE, pad=10)

            # Ensure the plot is displayed correctly
            # plt.tight_layout()
            plt.savefig(f'{path}/{str(raw["epoch"]).zfill(3)}_{stage_name}_{raw["modality"]}.svg',
                        format='svg', bbox_inches='tight')
            plt.close('all')
        # plt.figure(figsize=(8, 6), dpi=200)
        # sns.heatmap(cm, annot=True, fmt='.2%', cmap="Blues", xticklabels=display_labels[np.unique(labels)],
        #             yticklabels=display_labels[np.unique(labels)], annot_kws={"size": 16})
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.title('Confusion Matrix')


def plot_roc_curve(csv_logger, stage_name='test', path=None, n_classes=3):
    if path is None:
        path = os.path.dirname(csv_logger.dir_path)
    dataframe = csv_logger.dataframe()
    if not isinstance(stage_name, (list, tuple)):
        stage_name = (stage_name,)

    def save_plot(y_test, y_score, name):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure(figsize=(10, 8), dpi=200)
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.title("Average ROC Curve to multiclass", fontsize=20)
        plt.legend(loc="lower right")
        os.makedirs(os.path.join(path, 'ROC'), exist_ok=True)
        plt.savefig(os.path.join(path, 'ROC', f"ROC_{name}.png"))
        plt.close('all')

    for m in dataframe['modality'].drop_duplicates():
        dataframe_m = dataframe[dataframe['modality'] == m]
        for stage in stage_name:
            dataframe_m_s = dataframe_m[dataframe_m['stage_name'] == stage]
            for e in dataframe_m_s['epoch']:
                if e < 100:
                    continue
                dataframe_m_s_e = dataframe_m_s[dataframe_m_s['epoch'] == e]
                # B x numclass
                labels = np.eye(n_classes)[np.array(dataframe_m_s_e['labels'].to_list()).flatten()]
                scores = np.array(dataframe_m_s_e['scores'].to_list()).squeeze()
                save_plot(labels, scores, f'{m}_{stage}_{e}')


from .avtivation_maps import CAM, GradCAM
from torchvision.utils import make_grid


def compute_gradcam(model, inp, modality):
    inp = {modality: inp[modality]}
    target_layers = f'encoder.encoder.{modality}._bn1'
    gradcam = GradCAM(nn_module=model, target_layers=target_layers)
    heatmap_grad = gradcam(inp, modality=modality)
    return heatmap_grad


class ActPlot:
    def __init__(self, dir, save=False):
        self.save = save
        if save:
            self.dir = os.path.join(dir, 'ActMaps')
            os.makedirs(self.dir, exist_ok=True)

    def __call__(self, stage_name, modality, inp, label, model):
        if modality == 'MM':
            return
        heatmap_grad = compute_gradcam(model, inp, modality)
        img = inp[modality]
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4]).cpu()

        # fc_layers = f'classifier.classifier.classifier'
        # try:
        #     cam = CAM(nn_module=model, target_layers=target_layers, fc_layers=fc_layers)
        #     heatmap = cam(inp, modality=modality)
        #     self.plot_slices_cv2(img, heatmap.cpu(), name=f'{modality}_{m}_{stage_name}_cam')
        # except:
        #     print(f'No CAM for {modality} {m} {stage_name}')
        self.plot_slices_cv2(img, heatmap_grad.cpu(), name=f'{modality}_{stage_name}_{label[0].item()}_gradcam')

        # gradcam = GradCAM(nn_module=model, target_layers='_blocks.6')

    def plot_slices(self, inp, heatmap, name='test.png'):
        plt.figure(figsize=(8, 8), dpi=300)
        plt.axis("off")

        plt.imshow(np.transpose(
            make_grid(inp, nrow=3, padding=2, normalize=True),
            (1, 2, 0)),
            animated=True,
            cmap='viridis', alpha=1
        )
        plt.imshow(np.transpose(
            1 - make_grid(heatmap, nrow=3, padding=2, normalize=True),
            (1, 2, 0))[..., 0],
                   animated=True,
                   cmap='rainbow', alpha=.3
                   )
        # import pdb
        # pdb.set_trace()
        idx = 0
        new_name = name + f'_{idx:03d}.png'
        while os.path.exists(f'{self.dir}/{new_name}'):
            idx += 1
            new_name = name + f'_{idx:03d}.png'
        plt.savefig(f'{self.dir}/{new_name}')

    def plot_slices_cv2(self, inp, heatmap, name):
        idx = 0
        new_name = name + f'_{idx:03d}.png'
        while os.path.exists(f'{self.dir}/{new_name}'):
            idx += 1
            new_name = name + f'_{idx:03d}.png'
        inp = np.transpose(make_grid(inp, nrow=3, padding=2, normalize=True), (1, 2, 0))
        inp = np.array(inp * 255, dtype=np.uint8)

        heatmap = 1 - np.transpose(make_grid(heatmap, nrow=3, padding=2, normalize=True), (1, 2, 0))
        heatmap = np.array(heatmap * 255, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 仅叠加显著的 heatmap 部分
        # 定义蓝色在 BGR 格式中的范围
        # 这些值可能需要根据你的热图进行调整
        blue_lower_bound = np.array([128, 0, 0])  # 蓝色的低阈值
        blue_upper_bound = np.array([255, 50, 50])  # 蓝色的高阈值

        # 反转掩码：非蓝色区域为1，蓝色区域为0
        mask = cv2.bitwise_not(cv2.inRange(heatmap, blue_lower_bound, blue_upper_bound))
        red_map = heatmap * (np.expand_dims(mask, -1) > 0)

        cv2.imwrite(f'{self.dir}/{name.split(".")[0] + "heat.png"}', red_map)
        superimposed_img = cv2.addWeighted(red_map, .3, inp, .7, 0.)
        cv2.imwrite(f'{self.dir}/{new_name}', superimposed_img)
