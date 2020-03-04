#!/usr/bin/env python3

import argparse
import atexit
import functools
import os
import shutil
import sys
import tempfile
import traceback

# Rethinking:
# ResNet-56-A: 93.14 +- 0.12 base, 92.96 +- 0.26 after 40, 93.09 +- 0.14 after 160 * .1156411502
# ResNet-56-B: 93.14 +- 0.12 base, 92.54 +- 0.19 after 40, 93.05 +- 0.18 after 160 * .3791088785
# ResNet-110-A: 93.14 +- 0.24 base, 93.25 +- 0.29 after 40, 93.22 +- 0.22 after 160 * .1882836396
# ResNet-110-B: 93.14 +- 0.24 base, 92.89 +- 0.43 after 40, 93.60 +- 0.25 after 160 * .6272741807
# ResNet-34-A: 73.31 base, 72.77 after 40, 73.03 after = 90 * .1819645733
# ResNet-34-B: 73.31 base, 72.55 after 40, 72.91 after = 90 * .3154121864

def import_matplotlib():
    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(mpldir, 0o777 & ~umask)
    os.environ['MPLCONFIGDIR'] = mpldir
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.texmanager
    class TexManager(matplotlib.texmanager.TexManager):
        texcache = os.path.join(mpldir, 'tex.cache')

    matplotlib.texmanager.TexManager = TexManager
    pdfmpl = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 25,
        "font.size": 23,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 18,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "figure.figsize": (8, 4),
        "figure.dpi": 100,
        "legend.loc": 'best',
        'axes.titlepad': 20,
        'pdf.use14corefonts': True,
        'ps.useafm': True,
    }
    matplotlib.rcParams.update(pdfmpl)

assert 'matplotlib' not in sys.modules
import_matplotlib()

import common
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import inspect

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

def save_plot(func=None):
    if func:
        caller = func.__name__.split('_')[1:]
    else:
        caller = inspect.stack(1)[1].function.split('_')[1:]

    save_path = os.path.join(_DIRNAME, 'figures', *caller)
    os.makedirs(save_path, exist_ok=True)

    for figname in plt.get_fignums():
        fig = plt.figure(figname)
        plt.savefig(
            os.path.join(save_path, str(figname)) + '.pdf',
            dpi=100,
            pad_inches=0.05,
            bbox_inches='tight',
        )
        plt.close(fig)


def plot_resnet20_sparse_oneshot_lth():
    common.lth_plot(network=common.RESNET20, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet20_sparse_iterative_lth():
    common.lth_plot(network=common.RESNET20, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
    )
def plot_resnet20_sparse_iterative_lth_force():
    common.lth_plot(network=common.RESNET20, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery', 'finetune', 'lottery'],
                    force_single=True
    )

def plot_resnet20_sparse_iterative_lth_force_all():
    common.lth_plot(network=common.RESNET20, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
                    force_single=True,
                    force_all_single=True,
    )


def plot_resnet56_sparse_oneshot_lth():
    common.lth_plot(network=common.RESNET56, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    nbins=3,
                    nybins=4,
    )

def plot_resnet56_sparse_iterative_lth():
    common.lth_plot(network=common.RESNET56, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
                    comparison_points=[(0.1497, -0.0006), (0.1, 0.0016), (.05, -.0065), (.03, -.0135)],
                    comparison_label=common.CARREIRA,
                    nybins=4,
    )
def plot_resnet56_sparse_iterative_lth_force():
    common.lth_plot(network=common.RESNET56, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery', 'finetune', 'lottery'],
                    comparison_points=[(0.1497, -0.0006), (0.1, 0.0016), (.05, -.0065), (.03, -.0135)],
                    comparison_label=common.CARREIRA,
                    nybins=4,
                    force_single=True,
    )
def plot_resnet56_sparse_iterative_lth_force_all():
    common.lth_plot(network=common.RESNET56, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
                    comparison_points=[(0.1497, -0.0006), (0.1, 0.0016), (.05, -.0065), (.03, -.0135)],
                    comparison_label=common.CARREIRA,
                    nybins=4,
                    force_single=True,
                    force_all_single=True,
    )

def plot_resnet56_structured_A_lth():
    common.lth_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.9209166901474594, 0.9309-0.9314)],
                    comparison_err=[0.0014],
                    comparison_label=common.RETHINKING,
                    nbins=5,
    )

def plot_resnet56_structured_B_lth():
    common.lth_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.8711749788672866, 0.9305-0.9314)],
                    comparison_err=[0.0018],
                    comparison_label=common.RETHINKING,
                    nbins=5,
    )

def plot_resnet110_sparse_oneshot_lth():
    common.lth_plot(network=common.RESNET110, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet110_sparse_iterative_lth():
    common.lth_plot(network=common.RESNET110, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery'])
def plot_resnet110_sparse_iterative_lth_force():
    common.lth_plot(network=common.RESNET110, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'finetune', 'lottery'], force_single=True)
def plot_resnet110_sparse_iterative_lth_force_all():
    common.lth_plot(network=common.RESNET110, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery'], force_single=True, force_all_single=True)

def plot_resnet110_structured_A_lth():
    common.lth_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                    min_max_x=(0.9783, 0.955),
                    comparison_points=[(0.9772632282872189, 0.9325-0.9314)],
                    comparison_err=[0.0022],
                    comparison_label=common.RETHINKING,
                    dont_plot_x=[2,3]
    )

def plot_resnet110_structured_B_lth():
    common.lth_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.6875922984758561, 0.9360-0.9314)],
                    comparison_err=[0.0025],
                    comparison_label=common.RETHINKING,
                    nbins=4,
    )

def plot_resnet34_structured_A_lth():
    common.lth_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.9246381154681216, .7303-.7331)],
                    comparison_label=common.RETHINKING,
    )

def plot_resnet34_structured_B_lth():
    common.lth_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.8937159746276857, .7291-.7331)],
                    comparison_label=common.RETHINKING,
    )

def plot_resnet50_sparse_oneshot_lth():
    common.lth_plot(network=common.RESNET50, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet50_sparse_iterative_lth():
    common.lth_plot(network=common.RESNET50, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    comparison_points=[
                        (0.195, -0.0002),
                    ],
                    comparison_label=common.AMC,
                    to_ignore=['reinit', 'lr_lottery'],
                    nbins=5,
                    nybins=4,
    )
def plot_resnet50_sparse_iterative_lth_force():
    common.lth_plot(network=common.RESNET50, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.195, -0.0002)],
                    comparison_label=common.AMC,
                    to_ignore=['reinit', 'lr_lottery', 'finetune', 'lottery'],
                    nbins=5,
                    nybins=4,
                    force_single=True,
    )
def plot_resnet50_sparse_iterative_lth_force_all():
    common.lth_plot(network=common.RESNET50, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01),
                    comparison_points=[(0.195, -0.0002)],
                    comparison_label=common.AMC,
                    to_ignore=['reinit', 'lr_lottery'],
                    nbins=5,
                    nybins=4,
                    force_single=True,
                    force_all_single=True,
    )

def plot_gnmt_sparse_oneshot_lth():
    common.lth_plot(network=common.GNMT, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5), min_max_x=(None, 0.05))

def plot_gnmt_sparse_iterative_lth():
    common.lth_plot(network=common.GNMT, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5), min_max_x=(None, 0.02815),
                    comparison_points=[
                        (0.2, 26.86-26.77),
                        (0.15, 26.52-26.77),
                        (0.1, 26.19-26.77),
                    ],
                    nbins=8,
                    comparison_label=common.ZHU_GUPTA,
                    to_ignore=['reinit', 'lr_lottery'],
    )


def plot_gnmt_sparse_iterative_lth_force():
    common.lth_plot(network=common.GNMT, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5), min_max_x=(None, 0.02815),
                    comparison_points=[
                        (0.2, 26.86-26.77),
                        (0.15, 26.52-26.77),
                        (0.1, 26.19-26.77),
                    ],
                    nbins=8,
                    comparison_label=common.ZHU_GUPTA,
                    to_ignore=['reinit', 'lr_lottery', 'finetune', 'lottery'],
                    force_single=True,
    )
def plot_gnmt_sparse_iterative_lth_force_all():
    common.lth_plot(network=common.GNMT, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5), min_max_x=(None, 0.02815),
                    comparison_points=[
                        (0.2, 26.86-26.77),
                        (0.15, 26.52-26.77),
                        (0.1, 26.19-26.77),
                    ],
                    nbins=8,
                    comparison_label=common.ZHU_GUPTA,
                    to_ignore=['reinit', 'lr_lottery'],
                    force_single=True,
                    force_all_single=True,
    )




def plot_resnet56_sparse_oneshot_lth_core():
    common.lth_plot(network=common.RESNET56, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01), to_ignore=['reinit', 'lr_lottery'],
                    nbins=4,
                    nybins=4,
    )

def plot_resnet56_structured_B_lth_core():
    common.lth_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
                    nbins=5,
                    nybins=4,
    )

def plot_resnet34_structured_A_lth_core():
    common.lth_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                    to_ignore=['reinit', 'lr_lottery'],
                    nbins=5,
                    nybins=4,
    )

def plot_resnet50_sparse_oneshot_lth_core():
    common.lth_plot(network=common.RESNET50, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01), to_ignore=['reinit', 'lr_lottery'])

def plot_gnmt_sparse_oneshot_lth_core():
    common.lth_plot(network=common.GNMT, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5), min_max_x=(None, 0.05), to_ignore=['reinit', 'lr_lottery'],nbins=8,)






def plot_resnet20_sparse_oneshot_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET20, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))
def plot_resnet20_sparse_iterative_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET20, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet56_sparse_oneshot_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))
def plot_resnet56_sparse_iterative_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet56_structured_A_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                                comparison_points=[(0, 0.9296-0.9314), (160 * .1156411502, 0.9309-0.9314)],
                                comparison_err=[0.0026, 0.0014],
                                comparison_label=common.RETHINKING,
    )

def plot_resnet56_structured_B_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                                comparison_points=[(0, 0.9254-0.9314), (160 * .3791088785, 0.9305-0.9314)],
                                comparison_err=[0.0019, 0.0018],
                                comparison_label=common.RETHINKING,
    )

def plot_resnet110_sparse_oneshot_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))
def plot_resnet110_sparse_iterative_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.03, 0.01))

def plot_resnet110_structured_A_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.03, 0.01),
                                comparison_points=[(0, 0.9325-0.9314), (160 * .1882836396, 0.9322-0.9314)],
                                comparison_err=[0.0029, 0.0022],
                                comparison_label=common.RETHINKING,
    )

def plot_resnet110_structured_B_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.03, 0.01),
                                comparison_points=[(0, 0.9289-0.9314), (160 * .6272741807, 0.9360-0.9314)],
                                comparison_err=[0.0043, 0.0025],
                                comparison_label=common.RETHINKING,
    )

def plot_resnet34_structured_A_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_A, min_max_y=(-0.05, 0.01),
                                comparison_points=[(0, .7277-.7331), (90 * .1819645733, .7303-.7331)],
                                comparison_label=common.RETHINKING,
                                nbins=5,
    )

def plot_resnet34_structured_B_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_B, min_max_y=(-0.05, 0.01),
                                comparison_points=[(0, .7255-.7331), (90 * .3154121864, .7291-.7331)],
                                comparison_label=common.RETHINKING,
)

def plot_resnet50_sparse_oneshot_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET50, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-0.05, 0.01))
def plot_resnet50_sparse_iterative_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.RESNET50, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-0.05, 0.01))

def plot_gnmt_sparse_oneshot_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.GNMT, is_iterative=False, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5))
def plot_gnmt_sparse_iterative_epoch_for_epoch():
    common.epoch_for_epoch_plot(network=common.GNMT, is_iterative=True, prune_method=common.UNSTRUCTURED, min_max_y=(-6, 1.5))




def plot_resnet20_sparse_oneshot_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET20, is_iterative=False, prune_method=common.UNSTRUCTURED,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet56_sparse_oneshot_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.UNSTRUCTURED,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'],
                                nbins=5,
                                nybins=4,
    )

def plot_resnet56_structured_A_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_A,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet56_structured_B_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET56, is_iterative=False, prune_method=common.STRUCTURED_B,
                                nbins=5,
                                nybins=4,
                                min_max_y=(-0.06, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet110_sparse_oneshot_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.UNSTRUCTURED,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet110_structured_A_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_A,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet110_structured_B_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET110, is_iterative=False, prune_method=common.STRUCTURED_B,
                                min_max_y=(-0.03, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet34_structured_A_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_A,
                                nbins=5,
                                nybins=5,
                                min_max_y=(-0.06, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet34_structured_B_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET34, is_iterative=False, prune_method=common.STRUCTURED_B,
                                nbins=5,
                                nybins=5,
                                min_max_y=(-0.05, 0.01), to_ignore=['lr_lottery', 'reinit'])

def plot_resnet50_sparse_oneshot_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.RESNET50, is_iterative=False, prune_method=common.UNSTRUCTURED,
                                min_max_y=(-0.06, 0.01), to_ignore=['lr_lottery', 'reinit'],
                                nbins=5,
                                nybins=5,
    )

def plot_gnmt_sparse_oneshot_epoch_for_epoch_core():
    common.epoch_for_epoch_plot(network=common.GNMT, is_iterative=False, prune_method=common.UNSTRUCTURED,
                                min_max_y=(-8, 1.5), to_ignore=['lr_lottery', 'reinit'],
                                nbins=6,
                                nybins=6,
    )



def plot_resnet20_flops_best():
    common.flop_plot(common.RESNET20, only_best=True, nbins=6, nybins=6)
def plot_resnet50_flops_best():
    common.flop_plot(common.RESNET50, only_best=True, nbins=6, nybins=6)
def plot_resnet56_flops_best():
    common.flop_plot(common.RESNET56, only_best=True, nbins=6, nybins=6)
def plot_resnet110_flops_best():
    common.flop_plot(common.RESNET110, only_best=True, nbins=6, nybins=6)



def do_plot_func(func):
    func()
    save_plot(func)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', '-n', type=int, default=None)
    parser.add_argument('functions', nargs='*')
    args = parser.parse_args()

    to_plot = [(fname,func) for (fname, func) in globals().items()
               if fname.startswith('plot_') and callable(func)]
    if args.functions:
        to_plot = [(fname, func) for (fname, func) in to_plot if
                   any(f in fname for f in args.functions)]

    with mp.Pool(processes=args.nproc) as pool:
        res = []

        for (_, func) in to_plot:
            res.append(pool.apply_async(do_plot_func, args=(func,)))

        for r, (fname, _) in zip(res, to_plot):
            print('Waiting {}'.format(fname))
            exc = r.get()
            if exc:
                traceback.print_exception(*exc)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
