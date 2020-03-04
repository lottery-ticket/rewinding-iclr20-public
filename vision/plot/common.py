import collections
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
import numpy as np
import operator
import os
import pickle

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

with open(os.path.join(_DIRNAME, 'bases_dataframe.pkl'), 'rb') as f:
    _BASE_DATA = pickle.load(f)
with open(os.path.join(_DIRNAME, 'pruned_dataframe.pkl'), 'rb') as f:
    _PRUNED_DATAFRAME = pickle.load(f)

##### NETWORKS #####

RESNET20 = 'resnet20'
RESNET34 = 'resnet34'
RESNET50 = 'resnet50'
RESNET56 = 'resnet56'
RESNET110 = 'resnet110'
VGG16 = 'vgg16_nofc'
GNMT = 'gnmt'
RETHINKING = r'Liu et al. (2019)'
ZHU_GUPTA = r'Zhu \& Gupta (2017)'
# YAMAMOTO = r'Yamamoto \& Maeno ()'
CARREIRA = r"Carreira-Perpi\~{n}\'{a}n \& Idelbayev (2018)"
AMC = r'He et al. (2018)'
SAFE_RANGE = r'Rewinding safe zone'

CIFAR_NETWORKS = (RESNET20, RESNET56, RESNET110, VGG16)
IMAGENET_NETWORKS = (RESNET34, RESNET50)


REWIND_NORMAL = 'lottery'
REWIND_LOW_LR = 'lr_lottery'
FINETUNE_HIGH_LR = 'lr_finetune'
FINETUNE_NORMAL = 'finetune'
REINITIALIZE = 'reinit'

UNSTRUCTURED = 'Unstructured'
STRUCTURED_A = 'Structured-A'
STRUCTURED_B = 'Structured-B'

##### PLOTTING #####

_EPS = 1e-6

MEDIAN_MAX = 'max'
MEAN_STD = 'mean'
RANGE_METHOD = MEDIAN_MAX
if RANGE_METHOD == MEDIAN_MAX:
    _FLAVOR = 'median $\pm$ min/max'
elif RANGE_METHOD == MEAN_STD:
    _FLAVOR = 'mean $\pm 1$ std'
else:
    raise ValueError(RANGE_METHOD)

TrialAggregate = collections.namedtuple('TrialAggregate', ['center', 'below', 'above'])
def aggregate_trials(trial_data, get=operator.attrgetter('accuracy')):
    assert len(trial_data) > 0
    vals = [get(p) for p in trial_data.values() if get(p) is not None]
    if len(vals) == 1:
        center = vals[0]
        below = 0
        above = 0
    elif RANGE_METHOD == MEDIAN_MAX:
        center = np.median(vals)
        below = center - np.min(vals)
        above = np.max(vals) - center
    elif RANGE_METHOD == MEAN_STD:
        center = np.mean(vals)
        below = above = np.std(vals)

    return TrialAggregate(center, below, above)


def ms_of_method(method):
    if FINETUNE_HIGH_LR in method:
        x = 1
    elif REWIND_LOW_LR in method:
        x = 1
    elif FINETUNE_NORMAL in method:
        x =  1
    elif REWIND_NORMAL in method:
        x = 1.5
    elif REINITIALIZE in method:
        x = 1.5
    else:
        raise ValueError(method)

    return x * 10


def fmt_of_method(method):
    if FINETUNE_HIGH_LR in method:
        return 'o-.'
    elif REWIND_LOW_LR in method:
        return 'h-.'
    elif FINETUNE_NORMAL in method:
        return 'D--'
    elif REWIND_NORMAL in method:
        return '*:'
    elif REINITIALIZE in method:
        return 'x-.'
    else:
        raise ValueError(method)

def label_of_network(network):
    return {
        RESNET20: 'ResNet-20',
        RESNET34: 'ResNet-34',
        RESNET50: 'ResNet-50',
        RESNET56: 'ResNet-56',
        RESNET110: 'ResNet-110',
        VGG16: 'VGG-16',
        GNMT: 'GNMT',
    }[network]

def color_of_method(name):
    def c(*col):
        return (col[0]/255, col[1]/255, col[2]/255, 1)
    # cmap = [
    #     (REWIND_LOW_LR, '#EE6352'),
    #     (FINETUNE_HIGH_LR, '#59CD90'),
    #     (REWIND_NORMAL, '#3FA7D6'),
    #     (FINETUNE_NORMAL, '#FAC05E'),
    #     (REINITIALIZE, '#F79D84'),
    # ]
    # return next(y for (x, y) in cmap if x in name)

    cmap = [
        (FINETUNE_HIGH_LR, (230, 159, 0)),
        (REWIND_NORMAL, (86, 180, 233)),
        (FINETUNE_NORMAL, (0, 158, 115)),
        (REWIND_LOW_LR, (240, 228, 66)),
        (REINITIALIZE, (0, 114, 178)),
    ]
    return next(c(*y) for (x, y) in cmap if x == name)

def order_of_method(method):
    if SAFE_RANGE in method:
        return -1
    elif REWIND_LOW_LR in method:
        return 3
    elif REWIND_NORMAL in method:
        return 1
    elif FINETUNE_HIGH_LR in method:
        return 0
    elif FINETUNE_NORMAL in method:
        return 2
    elif REINITIALIZE in method:
        return 4
    else:
        return 5

def long_label_of_method(method):
    if REWIND_LOW_LR in method:
        return 'Low-LR weight rewinding'
    elif REWIND_NORMAL in method:
        return r'Weight rewinding'
    elif FINETUNE_HIGH_LR in method:
        return r'Learning rate rewinding'
    elif FINETUNE_NORMAL in method:
        return 'Fine-tuning'
    elif REINITIALIZE in method:
        return 'Reinitializing'
    else:
        return method


def label_of_method(method):

    if REWIND_LOW_LR in method:
        return 'LT-LR'
    elif REWIND_NORMAL in method:
        return 'LT'
    elif FINETUNE_HIGH_LR in method:
        return 'FT+LR'
    elif FINETUNE_NORMAL in method:
        return 'FT'
    elif REINITIALIZE in method:
        return REINITIALIZE
    else:
        raise ValueError(method)

def inv_label_of_method(method):
    return {
        'LT-LR': REWIND_LOW_LR,
        'LT': REWIND_NORMAL,
        'FT+LR': FINETUNE_HIGH_LR,
        'FT': FINETUNE_NORMAL,
        REINITIALIZE: REINITIALIZE
    }.get(method, method)

class BigFixedLocator(matplotlib.ticker.Locator):
    def __init__(self, locs, nbins=None):
        self.locs = np.asarray(locs)
        self.nbins = max(nbins, 2) if nbins is not None else None
    def set_params(self, nbins=None):
        if nbins is not None:
            self.nbins = nbins
    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        if self.nbins is None:
            return self.locs
        step = max(int(np.ceil(len(self.locs) / self.nbins)), 1)
        ticks = self.locs[::step]
        for i in range(1, step):
            ticks1 = self.locs[i::step]
            if np.abs(ticks1).min() < np.abs(ticks).min():
                ticks = ticks1

        if self.locs[-1] != ticks[-1]:
            # if abs((ticks[-1] - self.locs[-1]) / (ticks[-1] - ticks[-2])) < 0.5:
            #     ticks = ticks[:-1]
            ticks = list(ticks) + list(self.locs[-1:])

        return self.raise_if_exceeds(ticks)

def get_major_locator(densities, nbins=10):
    return BigFixedLocator(densities, nbins=nbins)

def _format_times_density(x):
    return r'${:.2f}\times$'.format(1/x)

def get_density_formatter():
    class SparsityFormatter(matplotlib.ticker.PercentFormatter):
        def __init__(self, *args, flip=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.flip = flip

        def __call__(self, x, i=None):
            return _format_times_density(x)
            if self.flip:
                x = 1 - x
            return super().__call__(x, i)

    return SparsityFormatter(xmax=1, decimals=None)

def get_density_interpolation_formatter(values, logscale, flip):
    class SparsityFormatter(matplotlib.ticker.PercentFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, x, i=None):
            if logscale:
                into, from_ = np.log, np.exp
            else:
                into = from_ = lambda x: x
            x = from_(np.interp(x, np.arange(len(values)), into(values)))
            return _format_times_density(x)
            if flip:
                x = 1 - x
            return super().__call__(x, i)

    return SparsityFormatter(xmax=1, decimals=None)

def get_retrain_interpolation_formatter(values, logscale, flip):
    class SparsityFormatter(matplotlib.ticker.PercentFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, x, i=None):
            if logscale:
                into, from_ = np.log, np.exp
            else:
                into = from_ = lambda x: x
            x = from_(np.interp(x, np.arange(len(values)), into(values)))
            if flip:
                x = 1 - x
            return super().__call__(x, i)

    return SparsityFormatter(xmax=1, decimals=None)

def get_accuracy_formatter(norm=None, is_delta=False):
    class AccuracyFormatter(matplotlib.ticker.PercentFormatter):
        def __init__(self, *args, flip=True, is_pct=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.flip = flip

        def format_pct(self, x, display_range):
            x = self.convert_to_pct(x)
            if self.decimals is None:
                # conversion works because display_range is a difference
                scaled_range = self.convert_to_pct(display_range)
                if scaled_range <= 0:
                    decimals = 0
                else:
                    decimals = math.ceil(2.0 - math.log10(2.0 * scaled_range))
                    if decimals > 5:
                        decimals = 5
                    elif decimals < 0:
                        decimals = 0
            else:
                decimals = self.decimals
            s = '{{x:{}0.{{decimals}}f}}'.format('+' if (is_delta and abs(x) > 0) else '').format(x=x, decimals=int(decimals))

            return s + self.symbol

    return AccuracyFormatter(xmax=1, decimals=0)

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_base_accuracy(network):
    return _BASE_DATA[_BASE_DATA['network'] == network].test_acc.mean()

def latexify(fig_width=9, fig_height=4.5):
    matplotlib.rcParams.update({
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (fig_width, fig_height),
        "figure.dpi": 72,
        "legend.loc": 'best',
        'axes.titlepad': 20,
    })

def detexify():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def format_axes(ax, twinx=False):
    SPINE_COLOR = 'gray'

    if twinx:
        visible_spines = ['bottom', 'right']
        invisible_spines = ['top', 'left']
    else:
        visible_spines = ['bottom', 'left']
        invisible_spines = ['top', 'right']

    for spine in invisible_spines:
        ax.spines[spine].set_visible(False)

    for spine in visible_spines:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')

    if twinx:
        ax.yaxis.set_ticks_position('right')
    else:
        ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def flop_plot(network, min_max_y=None, to_ignore=['lr_lottery', 'reinit'], only_best=False, nbins=10, nybins=10, dont_plot_x=None):
    base_flops = _BASE_DATA[_BASE_DATA['network'] == network]
    base_flops = base_flops[~base_flops['flops'].isna()]['flops'].mean()

    data = _PRUNED_DATAFRAME[
        (_PRUNED_DATAFRAME['network'] == network) &
        (_PRUNED_DATAFRAME['is_iterative']) &
        (~_PRUNED_DATAFRAME['flops'].isna())
    ]

    all_retrain_methods = sorted(set(data.retrain_method), key=order_of_method)
    all_densities = sorted(set(data.density))[::-1]
    all_retrain_epochs = sorted(set(data.retrain_time))

    def fmt_ax(ax, ax2=None):
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Speedup')

        ax.set_xscale('log')
        # ax.set_yscale('log')

        if min_max_y:
            ax.set_ylim((min_max_y[0], min_max_y[1]))

        if ax2:
            ax2.set_ylim((.95, 1.3))

        ax.set_xlim((ax.get_xlim()[1], ax.get_xlim()[0]))

        if dont_plot_x:
            densities = [all_densities[i] for i in range(len(all_densities))
                         if i not in dont_plot_x]
        ax.xaxis.set_major_locator(get_major_locator(all_densities, nbins=nbins))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(get_density_formatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nybins, steps = [1, 2, 2.5, 5, 10]))
        ax.grid(True)

        ax.set_ylim(0.1, ax.get_ylim()[1])
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(r'{x:.2f}$\times$'))

        # ax.legend(loc='upper left')
        fig.set_tight_layout(True)
        format_axes(ax)


    if only_best:
        data = data.groupby(['retrain_method', 'density', 'retrain_time']).agg({'flops': ['median', 'min', 'max'], 'test_acc': ['median', 'min', 'max'], 'val_acc': ['median', 'min', 'max']})
        maxes = data.loc[data[('val_acc', 'median')].groupby(['retrain_method', 'density']).idxmax().values].sort_values('density', ascending=False)

        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()

        for retrain_method in all_retrain_methods:
            if retrain_method in to_ignore:
                continue

            flops = maxes.loc[retrain_method]['flops']

            speedup_median = base_flops / flops['median']
            speedup_max = base_flops / flops['min']
            speedup_min = base_flops / flops['max']

            ax.errorbar(
                all_densities, speedup_median,
                [speedup_median - speedup_min, speedup_max - speedup_median],
                fmt=fmt_of_method(retrain_method),
                label=label_of_method(retrain_method),
                color=color_of_method(retrain_method),
                capsize=5,
                ms=ms_of_method(retrain_method),
                capthick=3,
                lw=3,
            )

            ft_flops = maxes.loc['finetune']['flops']
            ft_median_flops = ft_flops['median'].values
            ft_min_flops = ft_flops['min'].values
            ft_max_flops = ft_flops['max'].values

            m_median = ft_median_flops / flops['median'].values

            ax2.errorbar(
                all_densities,
                m_median,
                [
                    m_median - ft_min_flops / flops['max'].values,
                    ft_max_flops / flops['min'].values - m_median,
                ],
                fmt=fmt_of_method(retrain_method),
                label=label_of_method(retrain_method),
                color=color_of_method(retrain_method),
                capsize=5,
                ms=ms_of_method(retrain_method),
                capthick=3,
                lw=3,
            )

        ax.set_title(r'{} best-validated FLOPs across densities'.format(
            label_of_network(network),
        ))

        ax2.set_title(r'{} speedup of best-validated FLOPs over fine-tuning'.format(
            label_of_network(network),
        ))

        fmt_ax(ax)
        fmt_ax(ax2)
        ax2.set_ylim((.90, 1.3))

    else:
        data = data.groupby(['retrain_method', 'retrain_time', 'density']).agg({'flops': ['median', 'min', 'max'], 'test_acc': ['median', 'min', 'max'], 'val_acc': ['median', 'min', 'max']}).sort_values('density', ascending=False)

        for retrain_epochs in all_retrain_epochs:
            fig, ax = plt.subplots(num=str(retrain_epochs))
            ax2 = plt.twinx()

            for retrain_method in all_retrain_methods:
                if retrain_method in to_ignore:
                    continue

                grouped = data.loc[(retrain_method, retrain_epochs)]
                density = grouped.index

                ax.errorbar(
                    density, grouped['flops']['median'],
                    [grouped['flops']['median'] - grouped['flops']['min'], grouped['flops']['max'] - grouped['flops']['median']],
                    fmt=fmt_of_method(retrain_method),
                    label=label_of_method(retrain_method),
                    color=color_of_method(retrain_method),
                    capsize=5,
                )

                ax2.plot(
                    density, data.loc[('finetune', retrain_epochs)]['flops']['median'] / grouped['flops']['median'],
                    # fmt=fmt_of_method(retrain_method),
                    # label=label_of_method(retrain_method),
                    # color=color_of_method(retrain_method),
                )

            ax.set_title(r'{} FLOPs across densities, {} re-train epochs'.format(
                label_of_network(network),
                retrain_epochs,
            ))

            fmt_ax(ax, ax2)



def lth_plot(network, is_iterative, prune_method, min_max_y=None, min_max_x=None, is_delta=True, comparison_points=None, comparison_err=None, comparison_label=None, to_ignore=None, separate_legend=True, nbins=4, nybins=4, only_retrain_epochs=None, dont_plot_x=None, yticks=None, force_single=False, force_all_single=False):
    data = _PRUNED_DATAFRAME[
        (_PRUNED_DATAFRAME['network'] == network) &
        (_PRUNED_DATAFRAME['is_iterative'] == is_iterative) &
        (_PRUNED_DATAFRAME['prune_method'] == prune_method)
    ]

    fig, ax = plt.subplots()

    last_density_index_above = None

    if comparison_points:
        if comparison_err:
            ax.errorbar(*zip(*comparison_points), comparison_err, label=comparison_label, color='k', fmt='o--', ms=10, zorder=10, capsize=5, capthick=3)
        else:
            ax.scatter(*zip(*comparison_points), label=comparison_label, s=100, color='k', zorder=10)

    densities = sorted(set(data.density))[::-1]

    for retrain_method in sorted(set(data.retrain_method), key=lambda x: order_of_method(x[0])):
        if to_ignore and retrain_method in to_ignore:
            continue

        retrain_method_data = data[data.retrain_method == retrain_method]

        center = []
        below = []
        above = []

        for density in densities:
            density_data = retrain_method_data[abs(retrain_method_data.density - density) < _EPS]

            if force_single:
                max_rt = density_data['retrain_time'].max()
                if retrain_method == FINETUNE_NORMAL or retrain_method == FINETUNE_HIGH_LR:
                    density_data = density_data[density_data['retrain_time'] == max_rt]
                elif retrain_method == REWIND_NORMAL:
                    density_data = density_data[density_data['retrain_time'] == density_data[density_data['retrain_time'] <= 0.9 * max_rt]['retrain_time'].max()]

            res = density_data.groupby('retrain_time').agg({'test_acc':['median', 'min', 'max'], 'val_acc': 'median'}).sort_values(('val_acc', 'median'), ascending=False)
            best = res.iloc[0].test_acc
            center.append(best['median'])
            above.append(best['max'] - best['median'])
            below.append(best['median'] - best['min'])

        center = np.array(center)
        if is_delta:
            center -= get_base_accuracy(network)

        m_label = label_of_method(retrain_method)
        if force_all_single:
            m_label = 'Re-training with {}'.format(long_label_of_method(retrain_method))
        elif force_single:
            m_label = 'Our pruning algorithm'

        ax.errorbar(
            densities, center, [below, above], fmt=fmt_of_method(retrain_method),
            label=m_label,
            color=color_of_method(retrain_method),
            capsize=5,
            ms=ms_of_method(retrain_method),
            capthick=3,
            lw=3,
        )

    # if is_delta:
    #     ax.plot(densities, [0 for _ in densities], '--', color=(0,0,0,0.3))
    # else:
    #     ax.plot(densities, [get_base_accuracy(network) for _ in densities], '--', color=(0,0,0,0.3))

    ax.grid(True)

    ax.set_title(r'{}{} {} {}'.format(
        '{} '.format('CIFAR-10' if network in CIFAR_NETWORKS else 'ImageNet' if network in IMAGENET_NETWORKS else 'WMT16' if network == GNMT else '???') if is_iterative else '',
        label_of_network(network),
        prune_method,
        ' (iterative)' if is_iterative else '',
    ))
    ax.set_xlabel('Compression ratio')
    ax.set_xscale('log')

    if network == GNMT:
        score = 'BLEU'
    else:
        score = 'Accuracy'

    if is_delta:
        ax.set_ylabel(r'$\Delta$ {}'.format(score))
    else:
        ax.set_ylabel('{}'.format(score.capitalize()))

    if min_max_y:
        ax.set_ylim((min_max_y[0], min_max_y[1]))

    if min_max_x:
        min_x, max_x = min_max_x
        if min_x is None:
            min_x = ax.get_xlim()[1]
        if max_x is None:
            max_x = ax.get_xlim()[0]
        ax.set_xlim((min_x, max_x))
    else:
        ax.set_xlim((ax.get_xlim()[1], ax.get_xlim()[0]))

    if dont_plot_x:
        densities = [densities[i] for i in range(len(densities)) if i not in dont_plot_x]
    ax.xaxis.set_major_locator(get_major_locator(densities, nbins=nbins))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(get_density_formatter())

    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nybins, steps = [1, 2, 2.5, 5, 10]))

    if network != GNMT:
        ax.yaxis.set_major_formatter(get_accuracy_formatter(is_delta=is_delta))

    (h, l) = ax.get_legend_handles_labels()
    sortidx = sorted(range(len(l)), key=lambda i: order_of_method(inv_label_of_method(l[i])))
    h_sort = [h[i] for i in sortidx]
    l_sort = [long_label_of_method(inv_label_of_method(l[i])) for i in sortidx]

    if force_single and comparison_label and separate_legend:
        ax.legend(h_sort[:], l_sort[:1], loc='lower left')
    elif comparison_label and separate_legend:
        ax.legend(h_sort[-1:], l_sort[-1:], loc='lower left')
    elif not separate_legend:
        ax.legend(h_sort, l_sort, loc='lower left')

    fig.set_tight_layout(True)
    format_axes(ax)

    if separate_legend:
        l_sort = [long_label_of_method(inv_label_of_method(l[i])) for i in sortidx]
        if force_single:
            h_sort = h_sort[1:]
            l_sort = l_sort[1:]
        elif comparison_label:
            h_sort = h_sort[:-1]
            l_sort = l_sort[:-1]

        legendfig = plt.figure()
        leg = legendfig.legend(h_sort, l_sort, 'center', fontsize=20, frameon=False)
        legendfig.set_size_inches(5, 2)
        legendfig.set_tight_layout(True)

        hlegendfig = plt.figure()
        leg = hlegendfig.legend(h_sort, l_sort, 'center', fontsize=20, frameon=False, ncol=2)
        hlegendfig.set_size_inches(11, 0.8)
        hlegendfig.set_tight_layout(True)
        plt.show()


def epoch_for_epoch_plot(network, is_iterative, prune_method, min_max_y=None, min_max_x=None, is_delta=True, to_ignore=None, separate_legend=True, nbins=10, plot_vertical_lines=False, comparison_points=None, comparison_err=None, comparison_label=None, nybins=4):
    data = _PRUNED_DATAFRAME[
        (_PRUNED_DATAFRAME['network'] == network) &
        (_PRUNED_DATAFRAME['is_iterative'] == is_iterative) &
        (_PRUNED_DATAFRAME['prune_method'] == prune_method)
    ]

    for density in sorted(set(data.density))[::-1]:
        fig, ax = plt.subplots(num=str(density))

        if network in CIFAR_NETWORKS:
            lines = [91, 136]
            maxep = 182
        elif network in IMAGENET_NETWORKS:
            lines = [30, 60, 80]
            maxep = 90
        elif network == GNMT:
            lines = [7.5, 8.33, 9.16]
            maxep = 5
        else:
            raise ValueError()

        if plot_vertical_lines:
            lines = np.array(lines)
            lines = maxep - lines

            lines = [maxep * 0.4, maxep * 0.9]

            for l in lines:
                ax.axvline(l, color=(0,0,0,0.3), ls='--')

        ax.fill_between([maxep*0.25, maxep*0.9], [-100, -100], [100, 100], facecolor='grey', alpha=0.2, label=SAFE_RANGE)

        for retrain_method in sorted(set(data.retrain_method), key=order_of_method):
            if to_ignore is not None and retrain_method in to_ignore:
                continue

            m_data = data[(data['retrain_method'] == retrain_method) & (abs(data['density'] - density) < _EPS)].groupby('retrain_time').agg({'test_acc': ['median', 'min', 'max']}).sort_values('retrain_time')

            xs = m_data.index
            center = m_data[('test_acc', 'median')]
            above = m_data[('test_acc', 'max')] - center
            below = center - m_data[('test_acc', 'min')]

            if is_delta:
                center -= get_base_accuracy(network)

            ax.errorbar(
                xs, center, [below, above], fmt=fmt_of_method(retrain_method),
                label=label_of_method(retrain_method),
                color=color_of_method(retrain_method),
                capsize=5,
                ms=ms_of_method(retrain_method),
                capthick=3,
                lw=3,
            )

            ax.set_title(r'{} {}, ${:.2f}\times${}'.format(
                label_of_network(network),
                prune_method,
                1/density,
                ' (iterative)' if is_iterative else '',
            ))
            ax.set_xlabel('Re-training epochs')
            if network == GNMT:
                score = 'BLEU'
            else:
                score = 'Accuracy'

            if is_delta:
                ax.set_ylabel(r'$\Delta$ {}'.format(score))
            else:
                ax.set_ylabel('{}'.format(score.capitalize()))

            if min_max_y:
                ax.set_ylim((min_max_y[0], min_max_y[1]))

            if min_max_x:
                min_x, max_x = min_max_x
                ax.set_xlim((min_x, max_x))

            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nybins, steps = [1, 2, 2.5, 5, 10]))
            ax_xs = list(xs)
            if network in CIFAR_NETWORKS:
                ax_xs = [10, 39, 67, 96, 125, 153, 182]
                nbins=None
            elif network in IMAGENET_NETWORKS:
                ax_xs = [9, 22, 36, 50, 63, 76, 90]
                nbins=None
            elif network == GNMT:
                ax_xs = [0.5, 1.25, 2, 2.75, 3.5, 4.25, 5]
                nbins=None

            ax.xaxis.set_major_locator(get_major_locator(ax_xs, nbins=nbins))

            if network != GNMT:
                ax.yaxis.set_major_formatter(get_accuracy_formatter(is_delta=is_delta))

        # if is_delta:
        #     ax.plot(xs, [0 for _ in xs], '--', color=(0,0,0,0.3))
        # else:
        #     ax.plot(xs, [get_base_accuracy(network) for _ in xs], '--', color=(0,0,0,0.3))

        ax.grid(True)

        has_comparison = False
        if comparison_points:
            if density != data['density'].max():
                continue

            has_comparison = True

            xs = [x[0] for x in comparison_points]
            ys = [x[1] for x in comparison_points]

            if comparison_err:
                ax.errorbar(xs, ys, comparison_err, color='k', label=comparison_label, fmt='o', ms=10, zorder=10, capsize=5, capthick=3,)
            else:
                ax.scatter(xs, ys, color='k', label=comparison_label, s=100, zorder=10)

        fig.set_tight_layout(True)
        format_axes(ax)

        (h, l) = ax.get_legend_handles_labels()
        sortidx = sorted(range(len(l)), key=lambda i: order_of_method(inv_label_of_method(l[i])))

        h_sort = [h[i] for i in sortidx]
        l_sort = [long_label_of_method(inv_label_of_method(l[i])) for i in sortidx]

        if has_comparison and separate_legend:
            ax.legend(h_sort[-1:], l_sort[-1:], loc='lower left')
        elif not separate_legend:
            ax.legend(h_sort, l_sort, loc='lower left')

    if separate_legend:
        legendfig = plt.figure()
        l_sort = [long_label_of_method(inv_label_of_method(l[i])) for i in sortidx]
        if comparison_label:
            h_sort = h_sort[:-1]
            l_sort = l_sort[:-1]

        if l_sort[0] == SAFE_RANGE:
            h_sort[0] = mpatches.Patch(color='gray', alpha=0.2, label='The red data')

            l_sort.append(l_sort.pop(0))
            h_sort.append(h_sort.pop(0))

        leg = legendfig.legend(h_sort, l_sort, 'center', fontsize=20, frameon=False)
        legendfig.set_size_inches(5, 2)
        legendfig.set_tight_layout(True)

        hlegendfig = plt.figure()
        if len(h_sort) == 1:
            ncol = 1
        elif len(h_sort) == 2:
            ncol = 2
        elif len(h_sort) == 3:
            ncol = 3
        elif len(h_sort) == 4:
            ncol = 4
        elif len(h_sort) > 4:
            ncol = 3

        leg = hlegendfig.legend(h_sort, l_sort, 'center', fontsize=20, frameon=False, ncol=ncol)
        hlegendfig.set_size_inches(11, 0.8)
        hlegendfig.set_tight_layout(True)
        plt.show()

SAFETY = 'safety'
DOMINANCE = 'dominance'
EPOCH_FOR_EPOCH = 'epoch for epoch'

def comparison_plot(data, topline_name, baseline_name, mode=EPOCH_FOR_EPOCH, plot_flops=False, cmap_lim=None, one_dim_plots=False):
    if plot_flops:
        agg = operator.attrgetter('flops')
    else:
        agg = operator.attrgetter('accuracy')

    topline = data.retrain_method_data[topline_name]
    baseline = data.retrain_method_data[baseline_name]

    assert len(topline.density_data) == len(baseline.density_data)
    assert all(len(d1) == len(d2) for d1 in topline.density_data.values() for d2 in baseline.density_data.values())

    n_densities = len(topline.density_data)
    n_retrains = len(next(iter(topline.density_data.values())).retrain_time_data)

    def get_data(data):
        table = np.empty((n_densities, n_retrains))
        for i, (density, density_data) in enumerate(sorted(data.density_data.items())[::-1]):
            for j, (retrain_time, retrain_time_data) in enumerate(sorted(density_data.retrain_time_data.items())):
                table[i, j] = aggregate_trials(retrain_time_data.trial_data, agg).center
        return table

    topline_data = get_data(topline)
    baseline_data = get_data(baseline)

    if mode == SAFETY:
        for i in range(1, n_retrains):
            baseline_data[:, i] = np.max([baseline_data[:, i - 1], baseline_data[:, i]], axis=0)
    elif mode == DOMINANCE:
        for i in range(n_retrains):
            baseline_data[:, i] = np.max(baseline_data, axis=1)

    if plot_flops:
        diff = topline_data / baseline_data
    else:
        diff = topline_data - baseline_data
    fig, ax = plt.subplots()

    if not cmap_lim:
        if network == GNMT:
            cmap_lim = 10
        else:
            cmap_lim = 0.01

    if plot_flops:
        midpoint = 1
    else:
        midpoint = 0

    norm = MidpointNormalize(midpoint=midpoint, vmin=midpoint-cmap_lim, vmax=midpoint+cmap_lim)
    cmap = matplotlib.cm.get_cmap('bwr')

    retrain_times = np.array(sorted(next(iter(topline.density_data.values())).retrain_time_data.keys())[::-1])

    if one_dim_plots:
        for i, retrain_time in enumerate(retrain_times):
            ax.plot(
                np.arange(len(diff[:, i])), diff[:, i],
                label='{} re-train epochs'.format(retrain_time),
            )
    else:
        im = ax.imshow(diff.T[::-1, :], cmap=cmap, norm=norm, aspect='auto')

    ax.set_title('{} comparison of {} {} against {}'.format(mode.capitalize(), label_of_network(data.network), label_of_method(topline_name), label_of_method(baseline_name)))

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
    ax.xaxis.set_major_formatter(get_density_interpolation_formatter(sorted(topline.density_data.keys())[::-1], logscale=True, flip=True))
    ax.set_xlabel('Compression ratio')

    if one_dim_plots:
        ax.legend()
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        # ax.yaxis.set_major_formatter(get_retrain_interpolation_formatter(retrain_times[::-1] / np.max(retrain_times), logscale=False, flip=False))
        # ax.set_ylabel('')
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        ax.yaxis.set_major_formatter(get_retrain_interpolation_formatter(retrain_times[::1] / np.max(retrain_times), logscale=False, flip=False))
        ax.set_ylabel('Re-training time')

    fig.set_tight_layout(True)
    format_axes(ax)

    if not one_dim_plots:
        fig2, ax2 = plt.subplots(figsize=(8, 1.5))
        cbar = fig.colorbar(im, cax=ax2, orientation='horizontal', format=get_accuracy_formatter(is_delta=not plot_flops))
        if plot_flops:
            label = 'FLOP ratio'
        else:
            label = r'$\Delta$ Accuracy'
        cbar.set_label(r'{} between {} and {}'.format(label, label_of_method(topline_name), label_of_method(baseline_name)))
        fig2.set_tight_layout(True)
        format_axes(ax2)
