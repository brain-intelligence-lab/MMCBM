import os
import seaborn as sns
import matplotlib.pyplot as plt


def line_plot(melt, save_path, format='svg', kwargs=None):
    default_kwargs = dict(
        font_scale=1.2,
        rc=None,
        palette='pastel',
        dpi=300,
        x='rs',
        xlabel='# of reports',
        ylabel='Performance score',
        y='score',
        hue='metrics',
        style=None,
        legend=True,
        legend_kws={},
    )
    if kwargs is not None:
        default_kwargs.update(kwargs)
    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1)
    sns.set_palette("pastel")
    plt.figure(figsize=(3.5, 3), dpi=300)
    ax = sns.lineplot(data=melt, x=default_kwargs['x'], y=default_kwargs['y'],
                      hue=default_kwargs['hue'], marker=True, dashes=False,
                      style=default_kwargs['style'], legend=default_kwargs['legend'], )
    # Aesthetics
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlabel(xlabel=default_kwargs['xlabel'])  # Add x-axis label
    plt.ylabel(ylabel=default_kwargs['ylabel'])  # Add y-axis label
    # plt.ylim(0.3, 1)  # 设置 y 轴的上下限为 0 和 10
    plt.legend(**default_kwargs['legend_kws'])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format=format, bbox_inches='tight', transparent=True)


def line_plot_grid(df, save_path, kwargs=None):
    # legend_kws=dict(loc='center', ncol=4, labelspacing=0.1, columnspacing=1.2, handlelength=1.2, frameon=False)
    default_kwargs = dict(
        font_scale=1.2,
        rc=None,
        palette='pastel',
        dpi=300,
        col='metrics',
        row=None,
        row_order=None,
        col_order=None,
        height=2.5,
        aspect=1.2,
        sharey=False,
        sharex=False,
        sharelabel=False,
        col_wrap=None,
        func=sns.lineplot,
        x='modality',
        y='score',
        xlim=None,
        ylim=None,
        hue='modality',
        hue_order=None,
        style=None,
        markers=True,
        dashes=False,
        legend=True,
        legend_kws={},
        legend_func=lambda x: x,

        title=None,
        xlabel=None,
        format='svg',
        xticklabels={},
        annotate=False,
        ci_height=0.07,
        anno_xytext=(0, 20),
        anno_fontsize=9,
        subplots_adjust=None,
    )
    if kwargs is not None:
        default_kwargs.update(kwargs)

    plt.rcParams['font.family'] = 'Arial'  # Arial or Helvetica
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=default_kwargs['font_scale'], rc=default_kwargs['rc'])
    sns.set_palette(default_kwargs['palette'])
    plt.figure(dpi=default_kwargs['dpi'])

    g = sns.FacetGrid(data=df, col=default_kwargs['col'], row=default_kwargs['row'],
                      row_order=default_kwargs['row_order'],  col_order=default_kwargs['col_order'],
                      height=default_kwargs['height'],
                      aspect=default_kwargs['aspect'], sharey=default_kwargs['sharey'],
                      sharex=default_kwargs['sharex'], col_wrap=default_kwargs['col_wrap'],
                      )
    g.map_dataframe(func=default_kwargs['func'], x=default_kwargs['x'], y=default_kwargs['y'],
                    hue=default_kwargs['hue'], hue_order=default_kwargs['hue_order'],
                    legend=default_kwargs['legend'], style=default_kwargs['style'],
                    markers=default_kwargs['markers'], dashes=default_kwargs['dashes'],
                    palette=default_kwargs['palette'])

    # Annotating the bars with CI values
    if default_kwargs['annotate']:
        for ax in g.axes.flat:
            for patch in ax.patches:
                if patch.get_height() < 0.1:
                    continue
                ax.annotate(f'{patch.get_height():.2f}', (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                            ha='center', va='center', xytext=default_kwargs['anno_xytext'],
                            textcoords='offset points', fontsize=default_kwargs['anno_fontsize'])

    # Iterate over each axis to set the y-axis label as the title
    for idx, ax in enumerate(g.axes.flatten()):
        if default_kwargs['row']:
            title = g.row_names[idx // g._ncol]
        else:
            title = g.col_names[idx % g._ncol]
        if default_kwargs['sharelabel'] and idx % g._ncol != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(title.capitalize())

    g.set(title=default_kwargs['title'],
          xlabel=default_kwargs['xlabel'],
          xlim=default_kwargs['xlim'],
          ylim=default_kwargs['ylim'],
          xticks=default_kwargs['xticklabels']
          )
    # g.set(**default_kwargs['xticklabels'])
    legend_keys = list(df[default_kwargs['hue']].unique())
    if default_kwargs['style'] is not None:
        legend_keys.extend(list(df[default_kwargs['style']].unique()))
    legend_data = {k: v for k, v in g._legend_data.items() if k in legend_keys}
    legend_data = default_kwargs['legend_func'](legend_data)
    # label_order
    g.add_legend(legend_data=legend_data, **default_kwargs['legend_kws'])

    g.tight_layout()
    if default_kwargs['subplots_adjust'] is not None:
        g.fig.subplots_adjust(wspace=default_kwargs['subplots_adjust'][0], hspace=default_kwargs['subplots_adjust'][1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g.savefig(save_path, format=default_kwargs['format'], bbox_inches='tight', transparent=True)


def concept_number_plot(df, modality, save_path, format='svg'):
    MM = df[(df['modality'] == modality) & (df['rs'] == 1.0) & (df['stage_name'] == 'test')]
    MM = MM.drop(columns=['rs', 'cs', 'rn'])
    melt = MM.melt(id_vars=['cn', 'modality'], var_name='metric', value_name='score')
    line_plot(melt, save_path, format=format, kwargs=dict(
        x='cn',
        xlabel='# concepts',
        legend_kws=dict(
            loc='lower right'
        )
    ))


def report_number_plot(df, modality, save_path, format='svg'):
    MM = df[(df['modality'] == modality) & (df['cs'] == 1.0) & (df['stage_name'] == 'test')]
    MM = MM.drop(columns=['rs', 'cs', 'cn'])
    melt = MM.melt(id_vars=['rn', 'modality'], var_name='metric', value_name='score')
    line_plot(melt, save_path, format=format, kwargs=dict(
        x='rn',
        xlabel='# reports',
        legend_kws=dict(
            loc='lower right'
        )
    ))
