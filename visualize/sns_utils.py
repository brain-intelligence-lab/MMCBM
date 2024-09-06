import numpy as np
import seaborn as sns


def bootstrap_ci(x):
    n_boot = 1000  # Number of bootstrap iterations
    bootstrap_means = sns.algorithms.bootstrap(x, func=lambda y: y.mean(), n_boot=n_boot)

    # Compute confidence interval
    alpha = 0.05  # 95% confidence interval
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower, upper


def annotate_grid(g, df, x, cols='metrics', ci_height='auto', ci_fontsize=8, anno_height='auto',
                  anno_fontsize=9, place_num=3):
    # Annotating the bars with CI values
    for ax in g.axes.flat:
        legend = ax.get_legend_handles_labels()[-1]
        group_num = len(legend)
        x_labels = [text.get_text() for text in ax.get_xticklabels()]
        hue = None
        if legend[0] not in x_labels:
            for c in df.columns:
                for l in legend:
                    if l in df[c].unique():
                        hue = c
                        break
        else:
            group_num = 1
        all_patches = [patch for patch in sorted(ax.patches, key=lambda x: x.get_x()) if patch.get_height() > 0.01]
        y_lim = ax.get_ylim()[1] - ax.get_ylim()[0]
        for idx, patch in enumerate(all_patches):
            # Get the current modality
            x_names = x_labels[idx // group_num]
            # Filter data for this modality and metric
            title = ax.get_title().split(' = ')[-1]
            if hue is not None:
                data = df[(df[x].astype(str) == x_names) & (df[cols] == title) & (df[hue] == legend[idx % group_num])]['score']
            else:
                data = df[(df[x].astype(str) == x_names) & (df[cols] == title)]['score']
            # Calculate 95% CI using bootstrapping
            ci_lower, ci_upper = bootstrap_ci(data)
            x_pos = patch.get_x() + patch.get_width() / 2.
            y_value = patch.get_height()
            # Annotate the bar with CI

            if ci_height == 'auto' or ci_height is None:
                _ci_height = ci_upper + y_lim * 0.02
            else:
                _ci_height = y_value + ci_height

            if ci_height is not None:

                ax.text(x_pos, _ci_height, f'({ci_lower:.{place_num}f}, {ci_upper:.{place_num}f})',
                        ha="center", va="bottom", fontsize=ci_fontsize)

            if anno_height == 'auto' and ci_height is not None:
                _anno_height = _ci_height + y_lim * 0.07
            elif isinstance(anno_height, int) or isinstance(anno_height, float):
                _anno_height = _ci_height + anno_height
            else:
                _anno_height = _ci_height

            ax.text(x_pos, _anno_height, f'{y_value:.{place_num}f}',
                    ha="center", va="bottom", fontsize=anno_fontsize)
            #
            # ax.annotate(f'{patch.get_height():.2f}', (patch.get_x() + patch.get_width() / 2., patch.get_height()),
            #             ha='center', va='center', xytext=default_kwargs['anno_xytext'],
            #             textcoords='offset points', fontsize=default_kwargs['anno_fontsize'], )
