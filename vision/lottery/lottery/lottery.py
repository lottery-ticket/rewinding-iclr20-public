from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import tensorflow as tf
import os
from . import prune_functions
import numpy as np

_MASK_SUFFIX = '_m'
_WEIGHT_SUFFIX = '_w'
_BIAS_SUFFIX = '_b'

def weight_name_of_base_name(base_name):
    return base_name + _WEIGHT_SUFFIX

def mask_name_of_base_name(base_name):
    return base_name + _MASK_SUFFIX

def bias_name_of_base_name(base_name):
    return base_name + _BIAS_SUFFIX

def weight_name_of_mask_name(mask_name):
    name = mask_name[:-len(_MASK_SUFFIX)]
    return name + _WEIGHT_SUFFIX

def base_name_of_mask_name(mask_name):
    return mask_name[:-len(_MASK_SUFFIX)].split('/')[-1]

def is_mask_name(mask_name):
    return mask_name.endswith(_MASK_SUFFIX)

class CheckpointerHook(tf.train.SessionRunHook, ABC):
    def __init__(self, save_dir, checkpoint_iters):
        super().__init__()
        self.save_dir = save_dir
        self.checkpoint_iters = set(checkpoint_iters)
        self.already_checkpointed_iters = set()
        self._global_step_tensor = tf.train.get_global_step()

    def _checkpoint(self, iteration, session):
        if iteration != 'final' and iteration in self.already_checkpointed_iters:
            return

        self.already_checkpointed_iters.add(iteration)
        tf.logging.info('Checkpointing at iteration {}'.format(iteration))

        tf.gfile.MakeDirs(self.save_dir)
        tf.get_collection(tf.GraphKeys.SAVERS)[0].save(
            session,
            os.path.join(self.save_dir, 'checkpoint_iter_{}'.format(iteration)),
            write_state=False,
            write_meta_graph=False,
        )

    def after_create_session(self, session, coordinator):
        iteration = session.run(tf.train.get_global_step())
        if iteration in self.checkpoint_iters:
            self._checkpoint(iteration, session)

    def after_run(self, run_context, run_values):
        # check if we should checkpoint
        global_step = run_context.session.run(tf.train.get_global_step())
        if global_step in self.checkpoint_iters:
            self._checkpoint(global_step, run_context.session)

    def end(self, session):
        self._checkpoint('final', session)

class PruningHook(tf.train.SessionRunHook, ABC):
    def __init__(self, results_dir, pruning_checkpoint, resetting_checkpoint, pruning_method, model_dir, reset_global_step_to, force_reprune):
        super().__init__()
        self.results_dir = results_dir
        self.pruning_checkpoint = pruning_checkpoint
        self.resetting_checkpoint = resetting_checkpoint
        self.prune = pruning_method
        self.model_dir = model_dir
        self.has_pruned = False
        self.reset_global_step_to = reset_global_step_to
        self.force_reprune = force_reprune

    def after_create_session(self, session, coordinator):
        if self.has_pruned:
            return

        self.has_pruned = True

        tf.logging.info('Pruning weights at {} with {} and resetting to {}'.format(
            self.pruning_checkpoint,
            self.prune,
            self.resetting_checkpoint,
        ))

        saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
        # get weights to value at end

        session.graph._unsafe_unfinalize()

        # find current checkpoint masks (to make sure we're not unnecessarily restarting)
        with tf.variable_scope('', reuse=True):
            current_ckpt_mask_values = session.run([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if is_mask_name(v.op.name)])

        if self.pruning_checkpoint == 'reinitialize':
            session.run(tf.initializers.global_variables())
        else:
            saver.restore(session, self.pruning_checkpoint)

        # find current values and masks of pruneable weights
        with tf.variable_scope('', reuse=True):
            masks = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if is_mask_name(v.op.name)]
            weights = [tf.get_variable(weight_name_of_mask_name(mask.op.name)) for mask in masks]
            names = [base_name_of_mask_name(mask.op.name) for mask in masks]
            weight_values, mask_values = session.run([weights, masks])

        # cast masks to float32s (they are bfloat16s on TPUs which numpy can't work with)
        mask_dtype = mask_values[0].dtype
        weight_values = [w.astype(np.float32) for w in weight_values]
        mask_values = [m.astype(np.float32) for m in mask_values]

        # actually prune
        new_mask_values = self.prune(names, weight_values, mask_values)

        # log density ratio
        density_ratio = sum(m.sum() for m in new_mask_values) / (sum(m.size for m in new_mask_values) or 1)

        if self.results_dir:
            tf.gfile.MakeDirs(self.results_dir)
            with tf.gfile.Open(os.path.join(self.results_dir, 'density_ratio'), 'w') as f:
                f.write(str(density_ratio))

        # convert masks back to original dtype
        new_mask_values = [m.astype(mask_dtype) for m in new_mask_values]

        if self.reset_global_step_to:
            saver.restore(session, self.reset_global_step_to)
            with tf.variable_scope('', reuse=True):
                step = session.run(tf.train.get_global_step())

        # ~rewind~
        if all(np.allclose(m1,m2) for (m1,m2) in zip(current_ckpt_mask_values, new_mask_values)) and not self.force_reprune:
            tf.logging.info('Abort! Already pruned, resetting to current checkpoint')
            saver.restore(session, tf.train.latest_checkpoint(self.model_dir))
        else:
            if self.resetting_checkpoint == 'reinitialize':
                session.run(tf.initializers.global_variables())
            else:
                saver.restore(session, self.resetting_checkpoint)


            if self.reset_global_step_to:
                with tf.variable_scope('', reuse=True):
                    session.run(tf.train.get_global_step().assign(step))

        # assign masks
        session.run([mask.assign(new_mask) for (mask, new_mask) in zip(masks, new_mask_values)])


        session.graph.finalize()


def add_flags(flags):
    flags.DEFINE_string('lottery_results_dir', default=None, help='Directory to save results to.')
    flags.DEFINE_string('lottery_checkpoint_iters', default=None, help='Iterations to checkpoint the model at.')
    flags.DEFINE_string('lottery_prune_at', default=None, help='Checkpoint to prune at ("reinitialize" to randomly prune)')
    flags.DEFINE_string('lottery_pruning_method', default=None, help='Pruning method to use (something in prune_functions)')
    flags.DEFINE_string('lottery_reset_to', default=None, help='Checkpoint to reset to ("reinitialize" to train from scratch)')
    flags.DEFINE_string('lottery_reset_global_step_to', default=None, help='Checkpoint to reset global step to (for rewinding just LR)')
    flags.DEFINE_float('lottery_force_learning_rate', default=None, help='Learning rate to force to set to (rather than a schedule; for fine-tuning)')
    flags.DEFINE_bool('lottery_force_reprune', default=False, help='Force re-prune')

def get_hooks(
        results_dir=None,
        prune_from_checkpoint=None,
        prune_to_checkpoint=None,
        reset_global_step_to=None,
        prune_method=None,
        checkpoint_iters=None,
        model_dir=None,
        force_reprune=None,
):
    hooks = []
    if prune_from_checkpoint and prune_to_checkpoint and prune_method:
        hooks.append(PruningHook(
            results_dir,
            prune_from_checkpoint,
            prune_to_checkpoint,
            prune_functions.get_prune_function_by_name(prune_method),
            model_dir,
            reset_global_step_to,
            force_reprune,
        ))
    if checkpoint_iters:
        hooks.append(CheckpointerHook(
            results_dir,
            set(checkpoint_iters),
          ))
    return hooks

def hooks_from_flags(params):
    from_ = params['lottery_prune_at'] or None
    method = params['lottery_pruning_method'] or None
    to_ = params['lottery_reset_to'] or None
    gs_to_ = params['lottery_reset_global_step_to'] or None
    results_dir = params['lottery_results_dir'] or None
    model_dir = params.get('model_dir') or 'execution_data'.join(params['lottery_results_dir'].rsplit('results', 1))
    force_reprune = params['lottery_force_reprune']

    if params['lottery_checkpoint_iters']:
        checkpoint_iters = set(map(int, params['lottery_checkpoint_iters'].split(',')))
    else:
        checkpoint_iters = None

    return get_hooks(
        results_dir=results_dir,
        prune_from_checkpoint=from_,
        prune_to_checkpoint=to_,
        reset_global_step_to=gs_to_,
        prune_method=method,
        checkpoint_iters=checkpoint_iters,
        model_dir=model_dir,
        force_reprune=force_reprune,
    )

def get_lr_tensor(params):
    if params['lottery_force_learning_rate'] is None:
        return None
    return tf.constant(params['lottery_force_learning_rate'], dtype=tf.float32)
