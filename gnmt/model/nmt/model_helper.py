# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utility functions for building models."""
from __future__ import print_function

import collections
import hashlib
import numpy as np
import tensorflow as tf

from utils import misc_utils as utils
from utils import vocab_utils

from lottery import lottery

class CellWrapper(tf.contrib.rnn.RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, global_step=None, seq_len=None):
    """Create a cell with added input, state, and/or output dropout.

    Mask paddings ahead of reversed sequence if seq_len is not None.
    """
    super(CellWrapper, self).__init__()
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._global_step = tf.stop_gradient(global_step)
    if seq_len is not None:
      self._seq_len = tf.stop_gradient(seq_len)
    else:
      self._seq_len = None

  @property
  def wrapped_cell(self):
    return self._cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return [
          self._cell.zero_state(batch_size, dtype),
          tf.zeros([batch_size, 1], tf.int32)
      ]

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""
    orig_inputs = inputs
    if self._input_keep_prob < 1:
      # When using functional_rnn, the forward pass will be recomputed in the
      # backprop. So we need to make the dropout layer deterministic between
      # farward and backward pass. So we use stateless random to make sure the
      # generated random number is deterministic with a given seed. We also
      # want the drop out to be random across different global steps and time
      # steps. So we put both of them to the seeds.
      seeds = tf.stop_gradient(
          tf.stack([
              tf.cast(self._global_step, tf.int32) + tf.reduce_sum(state[1]),
              int(hashlib.md5(
                  inputs.name.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
          ]))
      keep_prob = tf.convert_to_tensor(
          self._input_keep_prob, dtype=tf.float32, name="keep_prob")
      random_tensor = keep_prob + tf.contrib.stateless.stateless_random_uniform(
          inputs.get_shape(), seed=tf.cast(seeds, tf.int32), dtype=tf.float32)
      binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
      inputs = tf.div(inputs, tf.cast(keep_prob, inputs.dtype)) * binary_tensor

    output, new_state = self._cell(inputs, state[0], scope=scope)
    if self._seq_len is not None:
      seq_len = tf.reshape(self._seq_len, [-1])
      padding = tf.reshape(state[1], [-1]) < (tf.reduce_max(seq_len) - seq_len)
      output = tf.where(padding, orig_inputs, output)
      new_state = tf.contrib.rnn.LSTMStateTuple(
          tf.where(padding, state[0].c, new_state.c),
          tf.where(padding, state[0].h, new_state.h))
    return output, [new_state, state[1] + 1]


__all__ = [
    "get_initializer", "create_emb_for_encoder_and_decoder", "create_rnn_cell",
    "gradient_clip"
]

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000


def get_initializer(init_op, seed=None, init_weight=0):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(
        seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(
        seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


class ExtraArgs(collections.namedtuple(
    "ExtraArgs", ("single_cell_fn", "model_device_fn",
                  "attention_mechanism_fn", "encoder_emb_lookup_fn"))):
  pass


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def _get_embed_device(vocab_size):
  """Decide on which device to place an embed matrix given its vocab size."""
  if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"


def _create_pretrained_emb_from_txt(
    vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32,
    scope=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    vocab_file: Path to vocab file.
    embed_file: Path to a Glove formmated embedding txt file.
    num_trainable_tokens: Make the first n tokens in the vocab file as trainable
      variables. Default is 3, which is "<unk>", "<s>" and "</s>".
    dtype: data type.
    scope: tf scope name.

  Returns:
    pretrained embedding table variable.
  """
  vocab, _ = vocab_utils.load_vocab(vocab_file)
  trainable_tokens = vocab[:num_trainable_tokens]

  utils.print_out("# Using pretrained embedding: %s." % embed_file)
  utils.print_out("  with trainable tokens: ")

  emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
  for token in trainable_tokens:
    utils.print_out("    %s" % token)
    if token not in emb_dict:
      emb_dict[token] = [0.0] * emb_size

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
  emb_mat = tf.constant(emb_mat)
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    emb_mat_var = tf.get_variable(
        "emb_mat_var", [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)


def _create_or_load_embed(embed_name, vocab_file, embed_file,
                          vocab_size, embed_size, dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file:
    embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
  else:
    embedding = tf.get_variable(lottery.weight_name_of_base_name(embed_name), [vocab_size, embed_size], dtype)
    embedding_mask = tf.get_variable(lottery.mask_name_of_base_name(embed_name), [vocab_size, embed_size], dtype,
                                     trainable=False, initializer=tf.initializers.ones()
    )
    embedding = tf.math.multiply(embedding, embedding_mask)
  return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_enc_partitions=0,
                                       num_dec_partitions=0,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None,
                                       src_embed_file=None,
                                       tgt_embed_file=None,
                                       use_char_encode=False,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder.

  Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
      encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    num_enc_partitions: number of partitions used for the encoder's embedding
      vars.
    num_dec_partitions: number of partitions used for the decoder's embedding
      vars.
    src_vocab_file: A string. The source vocabulary file.
    tgt_vocab_file: A string. The target vocabulary file.
    src_embed_file: A string. The source embedding file.
    tgt_embed_file: A string. The target embedding file.
    use_char_encode: A boolean. If true, use char encoder.
    scope: VariableScope for the created subgraph. Default to "embedding".

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """
  if num_enc_partitions <= 1:
    enc_partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    enc_partitioner = tf.fixed_size_partitioner(num_enc_partitions)

  if num_dec_partitions <= 1:
    dec_partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    dec_partitioner = tf.fixed_size_partitioner(num_dec_partitions)

  if src_embed_file and enc_partitioner:
    raise ValueError(
        "Can't set num_enc_partitions > 1 when using pretrained encoder "
        "embedding")

  if tgt_embed_file and dec_partitioner:
    raise ValueError(
        "Can't set num_dec_partitions > 1 when using pretrained decdoer "
        "embedding")

  with tf.variable_scope(
      scope or "embeddings", dtype=dtype, partitioner=enc_partitioner) as scope:
    # Share embedding
    if share_vocab:
      if src_vocab_size != tgt_vocab_size:
        raise ValueError("Share embedding but different src/tgt vocab sizes"
                         " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
      assert src_embed_size == tgt_embed_size
      utils.print_out("# Use the same embedding for source and target")
      vocab_file = src_vocab_file or tgt_vocab_file
      embed_file = src_embed_file or tgt_embed_file

      embedding_encoder = _create_or_load_embed(
          "embedding_share", vocab_file, embed_file,
          src_vocab_size, src_embed_size, dtype)
      embedding_decoder = embedding_encoder
    else:
      if not use_char_encode:
        with tf.variable_scope("encoder", partitioner=enc_partitioner):
          embedding_encoder = _create_or_load_embed(
              "embedding_encoder", src_vocab_file, src_embed_file,
              src_vocab_size, src_embed_size, dtype)
      else:
        embedding_encoder = None

      with tf.variable_scope("decoder", partitioner=dec_partitioner):
        embedding_decoder = _create_or_load_embed(
            "embedding_decoder", tgt_vocab_file, tgt_embed_file,
            tgt_vocab_size, tgt_embed_size, dtype)

  return embedding_encoder, embedding_decoder


class MaskedLSTMCell(tf.contrib.rnn.BasicLSTMCell):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_variable(self, name, shape=None, **kwargs):
    variable = super().add_variable(lottery.weight_name_of_base_name(name), shape, **kwargs)

    mask = super().add_variable(lottery.mask_name_of_base_name(name), shape, trainable=False, initializer=tf.initializers.ones())
    return tf.math.multiply(variable, mask)

class MaskedGRUCell(tf.contrib.rnn.GRUCell):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_variable(self, name, shape=None, **kwargs):
    variable = super().add_variable(lottery.weight_name_of_base_name(name), shape, **kwargs)

    mask = super().add_variable(lottery.mask_name_of_base_name(name), shape, trainable=False, initializer=tf.initializers.ones())
    return tf.math.multiply(variable, mask)

class MaskedLayerNormLSTMCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_variable(self, name, shape=None, **kwargs):
    variable = super().add_variable(lottery.weight_name_of_base_name(name), shape, **kwargs)

    mask = super().add_variable(lottery.mask_name_of_base_name(name), shape, trainable=False, initializer=tf.initializers.ones())
    return tf.math.multiply(variable, mask)

class MaskedNasCell(tf.contrib.rnn.NASCell):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_variable(self, name, shape=None, **kwargs):
    variable = super().add_variable(lottery.weight_name_of_base_name(name), shape, **kwargs)

    mask = super().add_variable(lottery.mask_name_of_base_name(name), shape, trainable=False, initializer=tf.initializers.ones())
    return tf.math.multiply(variable, mask)



def _single_cell(unit_type,
                 num_units,
                 forget_bias,
                 dropout,
                 mode,
                 residual_connection=False,
                 residual_fn=None,
                 global_step=None,
                 fast_reverse=False,
                 seq_len=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = MaskedLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    utils.print_out("  GRU", new_line=False)
    single_cell = MaskedGRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                    new_line=False)
    single_cell = MaskedLayerNormLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas":
    utils.print_out("  NASCell", new_line=False)
    single_cell = MaskedNasCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0 or fast_reverse:
    single_cell = CellWrapper(
        cell=single_cell,
        input_keep_prob=(1.0 - dropout),
        global_step=global_step,
        seq_len=seq_len)
    utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                    new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  return single_cell


def _cell_list(unit_type,
               num_units,
               num_layers,
               num_residual_layers,
               forget_bias,
               dropout,
               mode,
               single_cell_fn=None,
               residual_fn=None,
               global_step=None,
               fast_reverse=False,
               seq_len=None):
  """Create a list of RNN cells."""
  if not single_cell_fn:
    single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        residual_fn=residual_fn,
        global_step=global_step,
        fast_reverse=fast_reverse,
        seq_len=seq_len)
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    num_residual_layers,
                    forget_bias,
                    dropout,
                    mode,
                    single_cell_fn=None,
                    global_step=None,
                    fast_reverse=False,
                    seq_len=None):
  """Create multi-layer RNN cell.

  Args:
    unit_type: string representing the unit type, i.e. "lstm".
    num_units: the depth of each unit.
    num_layers: number of cells.
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
      cells in the returned list will be wrapped with `ResidualWrapper`.
    forget_bias: the initial forget bias of the RNNCell(s).
    dropout: floating point value between 0.0 and 1.0:
      the probability of dropout.  this is ignored if `mode != TRAIN`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    single_cell_fn: allow for adding customized cell.
      When not specified, we default to model_helper._single_cell
    global_step: the global step tensor.
    fast_reverse: If true, needs CellWrapper to mask paddings ahead of reversed
      sequence.
    seq_len: the sequence length tensor.
  Returns:
    An `RNNCell` instance.
  """
  cell_list = _cell_list(
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=forget_bias,
      dropout=dropout,
      mode=mode,
      single_cell_fn=single_cell_fn,
      global_step=global_step,
      fast_reverse=fast_reverse,
      seq_len=seq_len)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)

  return clipped_gradients, gradient_norm


from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import common_shapes

class Dense(base_layer.Layer):
  def __init__(self, units, *, use_bias=False, dtype=None, **kwargs):
    super(Dense, self).__init__(**kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    if dtype is None or use_bias:
      raise ValueError()

    self.use_bias = use_bias
    self.m_dtype = dtype

  def call(self, inputs):
    if self.use_bias:
      raise ValueError()

    rank = common_shapes.rank(inputs)
    outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
    shape = inputs.get_shape().as_list()
    output_shape = shape[:-1] + [self.units]
    outputs.set_shape(output_shape)
    return outputs

  def add_weight(self, name, shape, dtype, initializer=None):

    variable = super().add_weight(
      lottery.weight_name_of_base_name(name), shape=shape,
      dtype=dtype)

    mask = super().add_weight(
      lottery.mask_name_of_base_name(name),
      shape=shape,
      trainable=False,
      initializer=tf.initializers.ones())

    return tf.math.multiply(variable, mask)

  def build(self, input_shape):
     self.kernel = self.add_weight(
       'kernel',
       shape=[input_shape[-1], self.units],
       dtype=self.m_dtype,
     )

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    return input_shape[:-1].concatenate(self.units)
