#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import tensorflow as tf

_DIRNAME = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def _get_path(basename, trial, version, other_name=None):
    path = os.path.join(os.path.join(os.environ.get('DATA_DIR', 'gs://renda'), 'gnmt_results'), version, basename)
    if other_name:
        path = os.path.join(path, other_name)
    path = os.path.join(path, trial)
    return path


def run(path, **kwargs):
    args = [
        'python3', 'nmt.py',
        '--data_dir={}'.format(os.environ['MLP_PATH_GCS_NMT']),
        '--tpu_name={}'.format(os.environ['MLP_TPU_NAME']),
        '--out_dir={}'.format(path),
        '--model_dir={}'.format(path),
        '--activation_dtype=bfloat16',
        '--batch_size=2048',
        '--num_examples_per_epoch=2699264',
        '--learning_rate=0.002',
        '--target_bleu=500',
        '--warmup_steps=200',
        '--share_vocab=False',
        '--decay_scheme=luong234',
        '--lottery_results_dir={}'.format(os.path.join(path, 'lottery')),
        '--lottery_checkpoint_iters=0,1318,2636,3954,5272,6590,7908,9226,10544,11862,13180',
    ]
    args += ['{}={}'.format(k, v) for (k, v) in kwargs.items()]
    print(' '.join(args + ['--test_year=15', '--examples_to_infer=2560', '--mode=train_and_eval',]))
    proc = subprocess.Popen(args + ['--test_year=15', '--examples_to_infer=2560', '--mode=train_and_eval'], cwd=os.path.join(_DIRNAME, 'nmt'))
    res = proc.wait()
    if res == 0:
        print(' '.join(args + ['--test_year=14', '--examples_to_infer=3003', '--mode=infer']))
        subprocess.Popen(args + ['--test_year=14', '--examples_to_infer=3003', '--mode=infer'], cwd=os.path.join(_DIRNAME, 'nmt')).wait()
    sys.exit(res)



def _create_initial_state_at_checkpoint(to_path, from_path, from_iter):
    if tf.gfile.Exists(to_path):
        return

    print('Copying graph.pbtxt')
    tf.gfile.Copy(os.path.join(from_path, 'graph.pbtxt'), os.path.join(to_path, 'graph.pbtxt'))

    to_copy = []
    for f in tf.gfile.ListDirectory(os.path.join(from_path, 'lottery')):
        if f.startswith('checkpoint_iter_{}.'.format(from_iter)):
            to_copy.append(f)

    for f in to_copy:
        print('Copying {}'.format(f))
        tf.gfile.Copy(os.path.join(from_path, 'lottery', f), os.path.join(to_path, f))

    with tf.gfile.Open(os.path.join(to_path, 'checkpoint'), 'w') as f:
        f.write('model_checkpoint_path: "checkpoint_iter_{}"\n'.format(from_iter))
        f.write('all_model_checkpoint_paths: "checkpoint_iter_{}"\n'.format(from_iter))

def train(trial, version):
    run(_get_path('base', trial, version))

def finetune(trial, version, retrain_epochs, density, base, iteration_count):
    path = _get_path('finetune', trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density))
    base = os.path.join(base, trial)
    _create_initial_state_at_checkpoint(path, base, 'final')

    run(path, **{
        '--lottery_pruning_method': 'prune_all_to_global_{}'.format(density),
        '--lottery_reset_to': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_prune_at': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_force_learning_rate': '0.00025',
        '--max_train_epochs': 10 + retrain_epochs * iteration_count,
    })

def lr_finetune(trial, version, retrain_epochs, density, base):
    rewind_point = 863 * (10 - retrain_epochs)

    path = _get_path('lr_finetune', trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density))
    base = os.path.join(base, trial)
    _create_initial_state_at_checkpoint(path, base, rewind_point)

    run(path, **{
        '--lottery_pruning_method': 'prune_all_to_global_{}'.format(density),
        '--lottery_reset_to': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_prune_at': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_reset_global_step_to': '{}/lottery/checkpoint_iter_{}'.format(base, rewind_point),
        '--max_train_epochs': 10,
    })

def lr_lottery(trial, version, retrain_epochs, density, base):
    rewind_point = 863 * (10 - retrain_epochs)

    path = _get_path('lr_lottery', trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density))
    base = os.path.join(base, trial)
    _create_initial_state_at_checkpoint(path, base, rewind_point)

    run(path, **{
        '--lottery_pruning_method': 'prune_all_to_global_{}'.format(density),
        '--lottery_prune_at': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_reset_to': '{}/lottery/checkpoint_iter_{}'.format(base, rewind_point),
        '--lottery_force_learning_rate': '0.00025',
        '--max_train_epochs': 10,
    })

def lottery(trial, version, retrain_epochs, density, base):
    rewind_point = 863 * (10 - retrain_epochs)

    path = _get_path('lottery', trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density))
    base = os.path.join(base, trial)
    _create_initial_state_at_checkpoint(path, base, rewind_point)

    run(path, **{
        '--lottery_pruning_method': 'prune_all_to_global_{}'.format(density),
        '--lottery_prune_at': '{}/lottery/checkpoint_iter_final'.format(base),
        '--lottery_reset_to': '{}/lottery/checkpoint_iter_{}'.format(base, rewind_point),
        '--max_train_epochs': 10,
    })

def reinit(trial, version, retrain_epochs, density, base):
    path = _get_path('reinit', trial, version, 'retrain_{}/density_{}'.format(retrain_epochs, density))
    base = os.path.join(base, trial)

    run(path, **{
        '--lottery_pruning_method': 'prune_all_to_global_{}'.format(density),
        '--lottery_reset_to': 'reinitialize',
        '--lottery_prune_at': '{}/lottery/checkpoint_iter_final'.format(base),
        '--max_train_epochs': 10+retrain_epochs,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--trial', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--retrain-epochs', type=int)
    parser.add_argument('--iteration-count', type=int, default=1)
    parser.add_argument('--density', type=float)
    parser.add_argument('--base-dir')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.trial, args.version)
    else:
        assert args.retrain_epochs and args.density and args.base_dir

        if args.mode == 'finetune':
            finetune(args.trial, args.version, args.retrain_epochs, args.density, args.base_dir, args.iteration_count)
        elif args.mode == 'lottery':
            lottery(args.trial, args.version, args.retrain_epochs, args.density, args.base_dir)
        elif args.mode == 'lr_finetune':
            lr_finetune(args.trial, args.version, args.retrain_epochs, args.density, args.base_dir)
        elif args.mode == 'lr_lottery':
            lr_lottery(args.trial, args.version, args.retrain_epochs, args.density, args.base_dir)
        elif args.mode == 'reinit':
            reinit(args.trial, args.version, args.retrain_epochs, args.density, args.base_dir)
        else:
            raise ValueError()

if __name__ == '__main__':
    main()
