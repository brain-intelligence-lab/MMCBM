import os
import seaborn as sns
import matplotlib.pyplot as plt
from .sns_utils import annotate_grid


def performance_plot(melt, save_path, format='svg'):
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1)
    sns.set_palette("pastel")
    plt.figure(figsize=(4, 2.5), dpi=300)
    ax = sns.barplot(data=melt, x='modality', y='score', hue='metrics', width=0.6,
                     errorbar=('ci', 95), capsize=.05, errwidth=1.5, errcolor='grey',
                     order=['FA', 'ICGA', 'US', 'MM'])

    # Legend adjustment
    plt.ylim(0.5, 1)
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title='Metrics', loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', format=format, transparent=True)


def performance_plot_grid(df, save_path, kwargs=None):
    # legend_kws=dict(loc='center', ncol=4, labelspacing=0.1, columnspacing=1.2, handlelength=1.2, frameon=False)
    default_kwargs = dict(
        row=None,
        col='metrics',
        col_wrap=None,
        row_order=None,
        col_order=None,
        font_scale=1.3,
        rc=None,
        palette='pastel',
        dpi=300,
        height=2.5,
        aspect=1.2,
        sharey=False,
        sharex=False,
        sharexlabel=False,
        sharexticklabel=False,
        shareylabel=False,
        func=sns.barplot,
        x='modality',
        y='score',
        hue='modality',
        width=.9,
        order=None,
        hue_order=None,
        xlim=None,
        ylim=None,
        legend=True,
        legend_kws={},
        legend_func=lambda x: x,
        errorbar=('ci', 95),
        capsize=.3,
        errwidth=1,
        errcolor='grey',
        title=None,
        format='svg',
        xlabel=None,
        xticklabels={},
        annotate=False,
        ci_height='auto',
        ci_fontsize=9,
        anno_height='auto',
        anno_fontsize=10,
        subplots_adjust=None,
        place_num=3,
        bar_args=None
    )
    if kwargs is not None:
        default_kwargs.update(kwargs)

    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=default_kwargs['font_scale'], rc=default_kwargs['rc'])
    sns.set_palette(default_kwargs['palette'])
    plt.figure(dpi=default_kwargs['dpi'])

    g = sns.FacetGrid(data=df, col=default_kwargs['col'], row=default_kwargs['row'],
                      height=default_kwargs['height'],
                      row_order=default_kwargs['row_order'], col_order=default_kwargs['col_order'],
                      aspect=default_kwargs['aspect'], sharey=default_kwargs['sharey'],
                      sharex=default_kwargs['sharex'], col_wrap=default_kwargs['col_wrap'],
                      )
    g.map_dataframe(func=default_kwargs['func'], x=default_kwargs['x'], y=default_kwargs['y'],
                    hue=default_kwargs['hue'], width=default_kwargs['width'],
                    order=default_kwargs['order'],
                    hue_order=default_kwargs['hue_order'], legend=default_kwargs['legend'],
                    errorbar=default_kwargs['errorbar'], capsize=default_kwargs['capsize'],
                    errwidth=default_kwargs['errwidth'], errcolor=default_kwargs['errcolor'],
                    palette=default_kwargs['palette'])
    if default_kwargs['annotate']:
        annotate_grid(g, df, x=default_kwargs['x'], cols=default_kwargs['col'],
                      ci_height=default_kwargs['ci_height'], ci_fontsize=default_kwargs['ci_fontsize'],
                      anno_height=default_kwargs['anno_height'], anno_fontsize=default_kwargs['anno_fontsize'],
                      place_num=default_kwargs['place_num'])

    # Iterate over each axis to set the xy-axis label as the title
    for idx, ax in enumerate(g.axes.flatten()):
        ticklabels = [t.get_text() for t in ax.get_xticklabels()]
        if default_kwargs['row']:
            col = g.row_names[idx // g._ncol]
        else:
            col = g.col_names[idx % len(g.col_names)]
        ax.set_ylabel(col.title())
        if default_kwargs['xlabel'] is not None:
            ax.set_xlabel(default_kwargs['xlabel'].title())
        else:
            ax.set_xlabel('')

        if default_kwargs['shareylabel']:
            if idx % g._ncol != 0:
                ax.set_ylabel('')
        if default_kwargs['sharexlabel']:
            if (idx // g._ncol) < g._nrow - 1:
                ax.set_xlabel('')
        if default_kwargs['sharexticklabel']:
            if (idx // g._ncol) < g._nrow - 1:
                ax.set_xticklabels([])
        patches = [patch for patch in sorted(ax.patches, key=lambda x: x.get_x()) if patch.get_height() > 0.01]
        for i, bar in enumerate(patches):
            if default_kwargs['bar_args'] is not None and ticklabels[i] in default_kwargs['bar_args'].keys():  # 示例条件，根据需要调整
                params = default_kwargs['bar_args'][ticklabels[i]]
                bar.set_hatch(params[1])
                # bar.set_edgecolor('black')  # 修改边框颜色
                bar.set_facecolor(params[0])  # 修改填充颜色

    g.set(title=default_kwargs['title'],
          xlim=default_kwargs['xlim'],
          ylim=default_kwargs['ylim'])
    g.set_xticklabels(**default_kwargs['xticklabels'])
    # g.set_titles(col_template="{col_name} patrons", row_template="{row_name}")
    legend_data = default_kwargs['legend_func'](g._legend_data)
    # label_order

    g.add_legend(legend_data=legend_data, **default_kwargs['legend_kws'])

    # 修改图例样式
    for text, patch in zip(g._legend.get_texts(), g._legend.get_patches()):
        label = text.get_text().replace('+', '\n+')
        if default_kwargs['bar_args'] is not None and label in default_kwargs['bar_args'].keys():  # 示例条件，根据需要调整
            params = default_kwargs['bar_args'][label]
            patch.set_facecolor(params[0])
            patch.set_hatch(params[1]+params[1])
        # 可以根据需要添加更多条件
    g.tight_layout()
    if default_kwargs['subplots_adjust'] is not None:
        g.fig.subplots_adjust(wspace=default_kwargs['subplots_adjust'][0],
                              hspace=default_kwargs['subplots_adjust'][1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g.savefig(save_path, format=default_kwargs['format'], bbox_inches='tight', transparent=True)
