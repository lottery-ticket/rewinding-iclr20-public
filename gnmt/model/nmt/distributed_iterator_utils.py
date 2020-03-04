# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build input pipelines that span TPU pods for optimal performance.

It's common to batch sequences according to their length. Unfortunately, a
naive scaling of such an input pipeline across a pod will result in each host
choosing the sequence length bucket independently. Concretely, host A may select
sequences of a short length, while host B may select sequences of a very long
length. Because every step involves a blocking all-reduce phase, host A must
wait for host B.

The input pipeline designed within synchronizes the hosts such that they all
select a sequence length bucket of the same length, resulting in up to 50%
performance improvements across large TPU pod slices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.python.data.ops import multi_device_iterator_ops
from mlperf_compliance import mlperf_log
import async_checkpoint
import estimator as nmt_estimator
import low_level_runner
from utils import vocab_utils


def create_train_runner(hparams, num_workers):
  params = {}
  steps_per_epoch = int(hparams.num_examples_per_epoch/hparams.batch_size)
  return low_level_runner.TrainLowLevelRunner(
      iterations=steps_per_epoch//2,
      hparams=hparams,
      per_host_v1=True)

  input_fn = DistributedPipeline(hparams, num_workers)
  runner.initialize(input_fn, params)
  mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
  runner.build_model(model_fn, params)
  return runner


def train_fn(hparams, num_workers):
  """Copy of train function from estimator.py."""
  # TODO: Merge improvements into the original.
  # pylint: disable=protected-access
  hparams.tgt_sos_id, hparams.tgt_eos_id = nmt_estimator._get_tgt_sos_eos_id(
      hparams)
  model_fn = nmt_estimator.make_model_fn(hparams)

  if hparams.use_tpu_low_level_api:
    runner = create_train_runner(hparams, num_workers)
    mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
    input_fn = DistributedPipeline(hparams, num_workers)
    runner.initialize(input_fn, {})
    runner.build_model(model_fn, {})
    runner.train(0, hparams.num_train_steps)
    return 0.0

  # cluster = tf.contrib.cluster_resolver.TPUClusterResolver(hparams.tpu_name)
  # cluster_spec = cluster.cluster_spec()
  # print('cluster_spec: %s' % cluster_spec)
  # num_workers = cluster_spec.num_tasks('tpu_worker')
  # print('num_workers: %s' % num_workers)

  pipeline = DistributedPipeline(hparams, num_workers)

  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=0)
  mlperf_log.gnmt_print(key=mlperf_log.INPUT_SIZE,
                        value=hparams.num_examples_per_epoch)

  if hparams.use_tpu:
    run_config = nmt_estimator._get_tpu_run_config(hparams, True)
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        use_tpu=hparams.use_tpu,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size,
        predict_batch_size=hparams.infer_batch_size)
  else:
    raise ValueError("Distributed input pipeline only supported on TPUs.")

  hooks = [pipeline]
  if hparams.use_async_checkpoint:
    hooks.append(
        async_checkpoint.AsyncCheckpointSaverHook(
            checkpoint_dir=hparams.out_dir,
            save_steps=int(
                hparams.num_examples_per_epoch / hparams.batch_size)))

  estimator.train(
      input_fn=pipeline, max_steps=hparams.num_train_steps, hooks=hooks)
  # Return value is not used
  return 0.0


def train_and_eval_with_low_level_api(hparams, num_workers):
  """Train and evaluation function."""
  # pylint: disable=protected-access
  hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2
  model_fn = nmt_estimator.make_model_fn(hparams)
  train_runner = create_train_runner(hparams, num_workers)
  eval_runner = nmt_estimator.create_eval_runner(hparams, model_fn)
  mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
  train_input_fn = DistributedPipeline(hparams, num_workers)
  train_runner.initialize(train_input_fn, {})
  train_runner.build_model(model_fn, {})

  eval_input_fn = nmt_estimator.make_input_fn(
      hparams, tf.contrib.learn.ModeKeys.INFER)
  params = {
      "infer_batch_size": int(hparams.infer_batch_size / hparams.num_shards)
  }
  eval_runner.initialize(eval_input_fn, params)
  eval_runner.build_model(model_fn, params)

  score = 0.0
  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET, value=hparams.target_bleu)
  current_step = 0
  for i in range(hparams.max_train_epochs):
    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=i)
    tf.logging.info("Start training epoch %d", i)
    mlperf_log.gnmt_print(
        key=mlperf_log.INPUT_SIZE, value=hparams.num_examples_per_epoch)

    steps_per_epoch = int(hparams.num_examples_per_epoch / hparams.batch_size)
    train_runner.train(current_step, steps_per_epoch//2)
    current_step += steps_per_epoch//2
    train_runner.train(current_step, steps_per_epoch//2)
    current_step += steps_per_epoch//2

    mlperf_log.gnmt_print(
        key=mlperf_log.TRAIN_CHECKPOINT, value=("Under " + hparams.out_dir))
    tf.logging.info("End training epoch %d", i)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_START)
    predictions = list(eval_runner.predict())
    score = nmt_estimator.get_metric(hparams, predictions, current_step)
    tf.logging.info("Score after epoch %d: %f", i, score)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                          value={"value": score, "epoch": i})
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP, value=i)
    # if score >= hparams.target_bleu:
    #   mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": True})
    #   return score

  mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": False})
  return score


def train_and_eval_fn(hparams, num_workers):
  """Train and evaluation function."""
  # pylint: disable=protected-access
  mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
  hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2
  model_fn = nmt_estimator.make_model_fn(hparams)
  pipeline = DistributedPipeline(hparams, num_workers)

  run_config = nmt_estimator._get_tpu_run_config(hparams)
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      use_tpu=hparams.use_tpu,
      train_batch_size=hparams.batch_size,
      eval_batch_size=hparams.batch_size,
      predict_batch_size=hparams.infer_batch_size)

  score = 0.0
  mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
  mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET, value=hparams.target_bleu)
  for i in range(hparams.max_train_epochs):
    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH, value=i)
    tf.logging.info("Start training epoch %d", i)
    mlperf_log.gnmt_print(
        key=mlperf_log.INPUT_SIZE, value=hparams.num_examples_per_epoch)
    steps_per_epoch = int(hparams.num_examples_per_epoch / hparams.batch_size)
    max_steps = steps_per_epoch * (i + 1)
    estimator.train(input_fn=pipeline, max_steps=max_steps, hooks=[pipeline])
    mlperf_log.gnmt_print(
        key=mlperf_log.TRAIN_CHECKPOINT, value=("Under " + hparams.out_dir))
    tf.logging.info("End training epoch %d", i)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_START)
    score = nmt_estimator.get_metric_from_estimator(hparams, estimator)
    tf.logging.info("Score after epoch %d: %f", i, score)
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                          value={"value": score, "epoch": i})
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP, value=i)
    if score >= hparams.target_bleu:
      mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": True})
      return score

  mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": False})
  return score


class DistributedPipeline(tf.train.SessionRunHook):
  """DistributedPipeline encapsulates constructing the distributed pipeline.

  We use a class because we need to construct the pipeline in a graph managed
  by [TPU]Estimator. As a result, we cannot pre-construct it using a normal
  function, as Estimator wants to manage the graph itself.

  We use a class because we need to capture the initializer and pass it to the
  train call to TPUEstimator while simultaneously passing ourselves as the input
  function.
  """

  def __init__(self, hparams, num_hosts):
    """Constructs a DistributedPipeline.

    Args:
      hparams: The hparams object for this model.
      num_hosts: The number of hosts in the slice of the TPU pod.

    Throws:
      ValueError: If the passed values are invalid.
    """
    self._hparams = hparams
    self._num_hosts = num_hosts
    self._iterator = None
    self._outputs = None
    global_batch_size = hparams.batch_size
    if global_batch_size % num_hosts != 0:
      raise ValueError(
          "global_batch_size (%s) must be a multiple of num_hosts (%s)" %
          (global_batch_size, num_hosts))

  def after_create_session(self, session, coord):
    del coord
    start = time.time()
    session.run(self._iterator.initializer)
    tf.logging.info("Initialized multi-host dataset iterators in %d seconds",
                    time.time() - start)

  def __call__(self, params):
    if not self._outputs:
      self._iterator = _make_distributed_pipeline(self._hparams,
                                                  self._num_hosts)
      self._outputs = self._iterator.get_next()

    if "context" in params:
      current_host = params["context"].current_input_fn_deployment()[1]
    elif "dataset_index" in params:
      current_host = params["dataset_index"]
    else:
      raise ValueError('Expect "context" or "dataset_index" in params.')

    return self._outputs[current_host]


def _make_distributed_pipeline(hparams, num_hosts):
  """Makes the distributed input pipeline.

  make_distributed_pipeline must be used in the PER_HOST_V1 configuration.

  Note: we return both the input function and the hook because
  MultiDeviceIterator is not compatible with Estimator / TPUEstimator.

  Args:
    hparams: The hyperparameters to use.
    num_hosts: The number of hosts we're running across.

  Returns:
    A MultiDeviceIterator.
  """
  # TODO: Merge with the original copy in iterator_utils.py.
  # pylint: disable=g-long-lambda,line-too-long
  global_batch_size = hparams.batch_size

  if global_batch_size % num_hosts != 0:
    raise ValueError(
        "global_batch_size (%s) must be a multiple of num_hosts (%s)" %
        (global_batch_size, num_hosts))

  # Optionally choose from `choose_buckets` buckets simultaneously.
  if hparams.choose_buckets:
    window_batch_size = int(global_batch_size / hparams.choose_buckets)
  else:
    window_batch_size = global_batch_size

  per_host_batch_size = global_batch_size / num_hosts

  output_buffer_size = global_batch_size * 100

  resolver = low_level_runner.get_resolver(hparams)
  assert resolver
  job_name = resolver.get_job_name() or "tpu_worker"

  with tf.device("/job:%s/task:0/cpu:0" % job_name):
    # From estimator.py
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_dataset = tf.data.TextLineDataset(src_file).prefetch(output_buffer_size)
    tgt_dataset = tf.data.TextLineDataset(tgt_file).prefetch(output_buffer_size)
    mlperf_log.gnmt_print(
        key=mlperf_log.INPUT_BATCH_SIZE, value=global_batch_size)
    mlperf_log.gnmt_print(
        key=mlperf_log.TRAIN_HP_MAX_SEQ_LEN, value=hparams.src_max_len)

    # Define local variables that are parameters in iterator_utils.make_input_fn
    sos = hparams.sos
    eos = hparams.eos
    random_seed = hparams.random_seed
    num_buckets = hparams.num_buckets
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
    num_parallel_calls = 100  # constant in iterator_utils.py
    skip_count = None  # constant in estimator.py
    reshuffle_each_iteration = True  # constant in estimator.py
    use_char_encode = hparams.use_char_encode
    filter_oversized_sequences = True  # constant in estimator.py

    # From iterator_utils.py
    if use_char_encode:
      src_eos_id = vocab_utils.EOS_CHAR_ID
    else:
      src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    mlperf_log.gnmt_print(key=mlperf_log.INPUT_SHARD, value=1)
    if skip_count is not None:
      src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    def map_fn_1(src, tgt):
      src = tf.string_split([src]).values
      tgt = tf.string_split([tgt]).values
      src_size = tf.size(src)
      tgt_size = tf.size(tgt)
      size_ok_bool = tf.logical_and(src_size > 0, tgt_size > 0)
      if filter_oversized_sequences:
        oversized = tf.logical_and(src_size < src_max_len,
                                   tgt_size < tgt_max_len)
        size_ok_bool = tf.logical_and(size_ok_bool, oversized)

      if src_max_len:
        src = src[:src_max_len]
      if tgt_max_len:
        tgt = tgt[:tgt_max_len]
      return (src, tgt, size_ok_bool)

    src_tgt_bool_dataset = src_tgt_dataset.map(
        map_fn_1, num_parallel_calls=num_parallel_calls)
    src_tgt_bool_dataset = src_tgt_bool_dataset.filter(
        lambda src, tgt, filter_bool: filter_bool)

    def map_fn_2(src, tgt, unused_filter_bool):
      if use_char_encode:
        src = tf.reshape(vocab_utils.tokens_to_bytes(src), [-1])
        tgt = tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)
      else:
        src = tf.cast(src_vocab_table.lookup(src), tf.int32)
        tgt = tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)

      # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
      tgt_in = tf.concat(([tgt_sos_id], tgt), 0)
      tgt_out = tf.concat((tgt, [tgt_eos_id]), 0)

      # Add in sequence lengths.
      if use_char_encode:
        src_len = tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN)
      else:
        src_len = tf.size(src)
      tgt_len = tf.size(tgt_in)
      return src, tgt_in, tgt_out, src_len, tgt_len

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    src_tgt_dataset = src_tgt_bool_dataset.map(
        map_fn_2, num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.cache()
    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration).repeat()

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
      return x.padded_batch(
          window_batch_size,
          # The first three entries are the source and target line rows;
          # these have unknown-length vectors.  The last two entries are
          # the source and target row sizes; these are scalars.
          padded_shapes=(
              tf.TensorShape([src_max_len]),  # src
              tf.TensorShape([tgt_max_len]),  # tgt_input
              tf.TensorShape([tgt_max_len]),  # tgt_output
              tf.TensorShape([]),  # src_len
              tf.TensorShape([])),  # tgt_len
          # Pad the source and target sequences with eos tokens.
          # (Though notice we don't generally need to do this since
          # later on we will be masking out calculations past the true sequence.
          padding_values=(
              src_eos_id,  # src
              tgt_eos_id,  # tgt_input
              tgt_eos_id,  # tgt_output
              0,  # src_len -- unused
              0),
          # For TPU, must set drop_remainder to True or batch size will be None
          drop_remainder=True)  # tgt_len -- unused

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      """Calculate bucket_width by maximum source sequence length."""
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10
      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    if num_buckets > 1:
      batched_dataset = src_tgt_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func,
              reduce_func=reduce_func,
              window_size=window_batch_size))
    else:
      batched_dataset = batching_func(src_tgt_dataset)

    batched_dataset = batched_dataset.map(
        lambda src, tgt_in, tgt_out, source_size, tgt_in_size: (
            {"source": src,
             "target_input": tgt_in,
             "target_output": tgt_out,
             "source_sequence_length": source_size,
             "target_sequence_length": tgt_in_size}))

    re_batched_dataset = batched_dataset.apply(tf.contrib.data.unbatch()).batch(
        int(per_host_batch_size), drop_remainder=True)

    output_devices = [
        "/job:%s/task:%d/cpu:0" % (job_name, i) for i in range(num_hosts)
    ]

    options = tf.data.Options()
    options.experimental_numa_aware = True
    options.experimental_filter_fusion = True
    options.experimental_map_and_filter_fusion = True
    re_batched_dataset = re_batched_dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset=re_batched_dataset,
        devices=output_devices,
        max_buffer_size=10,
        prefetch_buffer_size=10,
        source_device=("/job:%s/task:0/cpu:0" % job_name))

    return multi_device_iterator
