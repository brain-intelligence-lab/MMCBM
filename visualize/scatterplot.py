import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def embedding_scatterplot(df, save_path, format='svg'):
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("pastel")
    plt.figure(dpi=300)
    # ax = sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=c, s=20)
    g = sns.FacetGrid(df, col='name', height=2.5, aspect=1)
    g.map_dataframe(sns.scatterplot, x='prx', y='pry', hue='pathology', edgecolor="black", linewidth=0.2,
                    s=40, alpha=0.8)
    g.set(title=None, xlabel=None, ylabel=None)
    g.add_legend(bbox_to_anchor=(.5, 0),
                 ncol=4,
                 labelspacing=0.1,
                 columnspacing=1.2,
                 handlelength=1.2, frameon=False)
    # ax.legend(loc='lower right')
    g.tight_layout()
    g.savefig(save_path, format=format, bbox_inches='tight', transparent=True)


def cav_scatterplot(df, save_path, format='svg'):
    df['concept'] = df['concept'].map(lambda x: x[:20] + '...' if len(x) > 20 else x)
    # Apply font settings
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("Set2")
    plt.figure(figsize=(12, 4), dpi=300)

    ax = sns.scatterplot(x='concept', y='acc', hue='stage', edgecolor="black", linewidth=0.5,
                         data=df.sort_values(by='concept', ascending=True), s=50)
    plt.xticks(rotation=90, va='bottom')
    ax.tick_params(axis='x', which='major', pad=120)  # 增加pad值以向下移动标签
    plt.xlabel(None)
    plt.ylabel('Accuracy Score')
    ax.legend(loc='lower right')
    # 设置图表边距
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.3)
    # 使用plt.margins()调整x轴两端的空白
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig(f'{save_path}', format=format, bbox_inches='tight', transparent=True)
