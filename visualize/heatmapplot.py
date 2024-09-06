import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import PercentFormatter


def heatmap_plot(cm, save_path=None, title=None, xlabel=True, ylabel=True, color_bar=True):
    # Font Settings
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_palette('pastel')

    TICK_LABEL_SIZE = 11
    FONT_SIZE = 12
    LABEL_SIZE = 12
    TITLE_SIZE = 13
    DPI = 300
    PAD = 5

    # 绘制混淆矩阵图
    FIGURE_SIZE = (4.5, 3.7)
    # Using a context manager for temporarily setting the font scale for seaborn
    with sns.plotting_context(rc={"font.size": FONT_SIZE,
                                  "axes.labelsize": LABEL_SIZE,
                                  "xtick.labelsize": TICK_LABEL_SIZE,
                                  "ytick.labelsize": TICK_LABEL_SIZE}):
        plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        ax = sns.heatmap(cm, annot=True, fmt='.2%', cmap="Blues")
        if color_bar:
            # Set colorbar labels format to percentage
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_aspect('equal')  # Ensure the heatmap is square
        if xlabel:
            ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, labelpad=PAD)
        else:
            ax.set_xlabel('')
        if ylabel:
            ax.set_ylabel('True Label', fontsize=LABEL_SIZE, labelpad=PAD)
        else:
            ax.set_ylabel('')
        if title is not None:
            plt.title(title, fontsize=TITLE_SIZE, pad=PAD)

        # Ensure the plot is displayed correctly
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
        return ax


def heatmap_grid(cm, save_path=None, title=None, format='svg'):
    # Font Settings
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_palette('pastel')

    TICK_LABEL_SIZE = 13
    FONT_SIZE = 14
    LABEL_SIZE = 14
    TITLE_SIZE = 15
    DPI = 300
    PAD = 5

    # 绘制混淆矩阵图
    FIGURE_SIZE = (9, 8.3)  # Adjusted for 2x2 grid
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    with sns.plotting_context(rc={"font.size": FONT_SIZE,
                                  "axes.labelsize": LABEL_SIZE,
                                  "xtick.labelsize": TICK_LABEL_SIZE,
                                  "ytick.labelsize": TICK_LABEL_SIZE}):
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.15)  # 分配空间给 colorbar

        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]  # 创建4个子图
        for idx, ax in enumerate(axes):
            sns.heatmap(cm[idx], annot=True, fmt='.0%', cmap="Blues", ax=ax, cbar=False)  # 关闭子图的 colorbar
            if idx > 1:
                ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, labelpad=PAD)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if idx % 2 == 0:
                ax.set_ylabel('True Label', fontsize=LABEL_SIZE, labelpad=PAD)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if title is not None:
                ax.set_title(title[idx], fontsize=TITLE_SIZE, pad=PAD)

        # 绘制统一的 colorbar
        # 该热力图应该包含所有热力图数据的最小值和最大值
        cbar_ax = fig.add_subplot(gs[:, -1])  # 分配 colorbar 的位置
        vmin = min(matrix.min().min() for matrix in cm)
        vmax = max(matrix.max().max() for matrix in cm)
        aux_heatmap = sns.heatmap(cm[0], ax=cbar_ax, vmin=vmin, vmax=vmax, cmap="Blues", cbar=False)
        plt.delaxes(aux_heatmap)  # 隐藏这个辅助的热力图
        cbar_ax = fig.add_subplot(gs[:, -1])  # 分配 colorbar 的位置
        cbar = fig.colorbar(aux_heatmap.collections[0], cax=cbar_ax, format=PercentFormatter(1, decimals=0))
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        for spine in cbar_ax.spines.values():  # Remove colorbar border
            spine.set_visible(False)

        plt.tight_layout()  # Adjust layout
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=format, bbox_inches='tight', transparent=True)
