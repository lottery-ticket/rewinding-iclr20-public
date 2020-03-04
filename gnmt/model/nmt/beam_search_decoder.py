# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import tensorflow as tf

import decoder
from mlperf_compliance import mlperf_log

import attention_utils

__all__ = [
    "BeamSearchDecoderOutput",
    "BeamSearchDecoderState",
    "BeamSearchDecoder",
    "tile_batch",
]


class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState",
                           ("cell_state", "log_probs", "finished", "lengths",
                            "accumulated_attention_probs", "pred_ids"))):
  pass


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput",
                           ("scores", "predicted_ids", "parent_ids"))):
  pass


def _tile_batch(t, multiplier):
  """Core single-tensor implementation of tile_batch."""
  t = tf.convert_to_tensor(t, name="t")
  shape_t = tf.shape(t)
  if t.shape.ndims is None or t.shape.ndims < 1:
    raise ValueError("t must have statically known rank")
  tiling = [1] * (t.shape.ndims + 1)
  tiling[1] = multiplier
  tiled_static_batch_size = (
      t.shape[0].value * multiplier if t.shape[0].value is not None else None)
  tiled = tf.tile(tf.expand_dims(t, 1), tiling)
  tiled = tf.reshape(
      tiled, tf.concat(([shape_t[0] * multiplier], shape_t[1:]), 0))
  tiled.set_shape(
      tf.TensorShape([tiled_static_batch_size]).concatenate(
          t.shape[1:]))
  return tiled


def tile_batch(t, multiplier, name=None):
  """Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

  For each tensor t in a (possibly nested structure) of tensors,
  this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed of
  minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
  `[batch_size * multiplier, s0, s1, ...]` composed of minibatch entries
  `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
  `multiplier` times.

  Args:
    t: `Tensor` shaped `[batch_size, ...]`.
    multiplier: Python int.
    name: Name scope for any created operations.

  Returns:
    A (possibly nested structure of) `Tensor` shaped
    `[batch_size * multiplier, ...]`.

  Raises:
    ValueError: if tensor(s) `t` do not have a statically known rank or
    the rank is < 1.
  """
  flat_t = tf.contrib.framework.nest.flatten(t)
  with tf.name_scope(name, "tile_batch", flat_t + [multiplier]):
    return tf.contrib.framework.nest.map_structure(
        lambda t_: _tile_batch(t_, multiplier), t)


def _check_maybe(t):
  if t.shape.ndims is None:
    raise ValueError(
        "Expected tensor (%s) to have known rank, but ndims == None." % t)


class BeamSearchDecoder(decoder.Decoder):
  """BeamSearch sampling decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Meanwhile, with `AttentionWrapper`, coverage penalty is suggested to use
    when computing scores(https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the translation to cover all inputs.
  """

  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               max_tgt=None,
               length_penalty_weight=0.0,
               coverage_penalty_weight=0.0,
               reorder_tensor_arrays=True,
               dtype=tf.float32):
    """Initialize the BeamSearchDecoder.

    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.variable` to represent the
        weights for the dense layer to apply to the RNN output prior
        to storing the result or sampling.
      max_tgt: maximum prediction length.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
      coverage_penalty_weight: Float weight to penalize the coverage of source
        sentence. Disabled with 0.0.
      reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
        state will be reordered according to the beam search path. If the
        `TensorArray` can be reordered, the stacked form will be returned.
        Otherwise, the `TensorArray` will be returned as is. Set this flag to
        `False` if the cell state contains `TensorArray`s that are not amenable
        to reordering.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
      ValueError: If `start_tokens` is not a vector or
        `end_token` is not a scalar.
    """
    self._cell = cell
    self._output_layer = output_layer
    self._reorder_tensor_arrays = reorder_tensor_arrays

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: tf.cast(tf.nn.embedding_lookup(embedding, ids), dtype))

    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")

    self._batch_size = start_tokens.shape[0].value
    self._beam_width = beam_width
    self._max_tgt = max_tgt
    self._length_penalty_weight = length_penalty_weight
    self._coverage_penalty_weight = coverage_penalty_weight
    self._initial_cell_state = tf.contrib.framework.nest.map_structure(
        self._maybe_split_batch_beams, initial_state, self._cell.state_size)
    self._start_tokens = tf.tile(
        tf.expand_dims(self._start_tokens, 1), [1, self._beam_width])
    self._start_inputs = self._embedding_fn(self._start_tokens)

    self._finished = tf.one_hot(
        tf.zeros([self._batch_size], dtype=tf.int32),
        depth=self._beam_width,
        on_value=False,
        off_value=True,
        dtype=tf.bool)

  @property
  def batch_size(self):
    return self._batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = tf.contrib.framework.nest.map_structure(
          lambda s: tf.TensorShape([None]).concatenate(s), size)
      layer_output_shape = output_shape_with_unknown_batch[:-1].concatenate(
          self._output_layer.shape[-1])
      return tf.contrib.framework.nest.map_structure(
          lambda s: s[1:], layer_output_shape)

  @property
  def tracks_own_finished(self):
    """The BeamSearchDecoder shuffles its beams and their finished state.

    For this reason, it conflicts with the `dynamic_decode` function's
    tracking of finished states.  Setting this property to true avoids
    early stopping of decoding due to mismanagement of the finished state
    in `dynamic_decode`.

    Returns:
      `True`.
    """
    return True

  @property
  def output_size(self):
    # Return the cell output and the id
    return BeamSearchDecoderOutput(
        scores=tf.TensorShape([self._beam_width]),
        predicted_ids=tf.TensorShape([self._beam_width]),
        parent_ids=tf.TensorShape([self._beam_width]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = tf.contrib.framework.nest.flatten(self._initial_cell_state)[0].dtype
    return BeamSearchDecoderOutput(
        scores=tf.contrib.framework.nest.map_structure(
            lambda _: dtype, self._rnn_output_size()),
        predicted_ids=tf.int32,
        parent_ids=tf.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished, start_inputs = self._finished, self._start_inputs

    dtype = tf.contrib.framework.nest.flatten(self._initial_cell_state)[0].dtype
    log_probs = tf.one_hot(  # shape(batch_sz, beam_sz)
        tf.zeros([self._batch_size], dtype=tf.int32),
        depth=self._beam_width,
        on_value=tf.convert_to_tensor(0.0, dtype=dtype),
        off_value=tf.convert_to_tensor(-np.Inf, dtype=dtype),
        dtype=dtype)
    init_attention_probs = get_attention_probs(
        self._initial_cell_state, self._coverage_penalty_weight)
    if init_attention_probs is None:
      init_attention_probs = ()
    init_pred_ids = tf.fill([self._batch_size, self._beam_width, self._max_tgt], self._end_token)

    initial_state = BeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=log_probs,
        finished=finished,
        lengths=tf.zeros(
            [self._batch_size, self._beam_width], dtype=tf.int64),
        accumulated_attention_probs=init_attention_probs,
        pred_ids=init_pred_ids)

    return (finished, start_inputs, initial_state)



  def _merge_batch_beams(self, t, s=None):
    """Merges the tensor from a batch of beams into a batch by beams.

    More exactly, t is a tensor of dimension [batch_size, beam_width, s]. We
    reshape this into [batch_size*beam_width, s]

    Args:
      t: Tensor of dimension [batch_size, beam_width, s]
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size * beam_width, s].
    """
    if isinstance(s, tf.Tensor):
      s = tf.contrib.util.constant_value(s)
      if isinstance(s, tf.TensorShape):
        return s
      else:
        s = tf.TensorShape(s)
    else:
      s = tf.TensorShape(s)
    t_shape = tf.shape(t)
    batch_size_beam_width = self._batch_size * self._beam_width
    reshaped_t = tf.reshape(
        t,
        tf.concat(([self._batch_size * self._beam_width], t_shape[2:]), 0))
    reshaped_t.set_shape(
        (tf.TensorShape([batch_size_beam_width]).concatenate(s)))
    return reshaped_t

  def _split_batch_beams(self, t, s=None):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
    reshape this into [batch_size, beam_width, s]

    Args:
      t: Tensor of dimension [batch_size*beam_width, s].
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, s].

    Raises:
      ValueError: If, after reshaping, the new tensor is not shaped
        `[batch_size, beam_width, s]` (assuming batch_size and beam_width
        are known statically).
    """
    if isinstance(s, tf.Tensor):
      s = tf.TensorShape(tf.contrib.util.constant_value(s))
    else:
      s = tf.TensorShape(s)
    t_shape = tf.shape(t)
    reshaped_t = tf.reshape(
        t,
        tf.concat(([self._batch_size, self._beam_width], t_shape[1:]), 0))
    expected_reshaped_shape = tf.TensorShape(
        [self._batch_size, self._beam_width]).concatenate(s)
    if not reshaped_t.shape.is_compatible_with(expected_reshaped_shape):
      raise ValueError("Unexpected behavior when reshaping between beam width "
                       "and batch size.  The reshaped tensor has shape: %s.  "
                       "We expected it to have shape "
                       "(batch_size, beam_width, depth) == %s.  Perhaps you "
                       "forgot to create a zero_state with "
                       "batch_size=encoder_batch_size * beam_width?" %
                       (reshaped_t.shape, expected_reshaped_shape))
    reshaped_t.set_shape(expected_reshaped_shape)
    return reshaped_t

  def _maybe_split_batch_beams(self, t, s):
    """Maybe splits the tensor from a batch by beams into a batch of beams.

    We do this so that we can use nest and not run into problems with shapes.

    Args:
      t: `Tensor`, either scalar or shaped `[batch_size * beam_width] + s`.
      s: `Tensor`, Python int, or `TensorShape`.

    Returns:
      If `t` is a matrix or higher order tensor, then the return value is
      `t` reshaped to `[batch_size, beam_width] + s`.  Otherwise `t` is
      returned unchanged.

    Raises:
      ValueError: If the rank of `t` is not statically known.
    """
    if isinstance(t, tf.TensorArray):
      return t
    _check_maybe(t)
    if t.shape.ndims >= 1:
      return self._split_batch_beams(t, s)
    else:
      return t

  def _maybe_merge_batch_beams(self, t, s):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, `t` is a tensor of dimension `[batch_size * beam_width] + s`,
    then we reshape it to `[batch_size, beam_width] + s`.

    Args:
      t: `Tensor` of dimension `[batch_size * beam_width] + s`.
      s: `Tensor`, Python int, or `TensorShape`.

    Returns:
      A reshaped version of t with shape `[batch_size, beam_width] + s`.

    Raises:
      ValueError:  If the rank of `t` is not statically known.
    """
    if isinstance(t, tf.TensorArray):
      return t
    _check_maybe(t)
    if t.shape.ndims >= 2:
      return self._merge_batch_beams(t, s)
    else:
      return t

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight
    coverage_penalty_weight = self._coverage_penalty_weight

    with tf.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      inputs = tf.contrib.framework.nest.map_structure(
          lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
      cell_state = tf.contrib.framework.nest.map_structure(
          self._maybe_merge_batch_beams, cell_state, self._cell.state_size)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = tf.contrib.framework.nest.map_structure(
          lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
      next_cell_state = tf.contrib.framework.nest.map_structure(
          self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

      if self._output_layer is not None:
        output_shape = cell_outputs.shape.as_list()[:-1] + [
            self._output_layer.shape[-1]
        ]
        cell_outputs = tf.matmul(
            tf.reshape(cell_outputs, [-1, cell_outputs.shape.as_list()[-1]]),
            tf.cast(self._output_layer, cell_outputs.dtype))
        cell_outputs = tf.reshape(cell_outputs, output_shape)

      beam_search_output, beam_search_state = _beam_search_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight,
          coverage_penalty_weight=coverage_penalty_weight,
          max_tgt=self._max_tgt)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = tf.cond(
          tf.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return (beam_search_output, beam_search_state, next_inputs, finished)


def create_make_unique(inputs):
  if inputs.shape.ndims != 2:
    raise ValueError("Input of top_k_with_unique must be rank-2 "
                     "but got: %s" % inputs.shape)
  height = inputs.shape[0]
  width = inputs.shape[1]
  zeros = tf.zeros([height, width], dtype=tf.int32)

  # count_mask is used to mask away the low order bits to ensure that every
  # element is distinct.
  log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = ~(next_power_of_two - 1)
  count_mask_r0 = tf.constant(count_mask)
  count_mask_r2 = tf.fill([height, width], count_mask_r0)

  # smallest_normal is the bit representation of the smallest
  # positive normal floating point number. The sign is zero,
  # exponent is one, and the fraction is zero.
  smallest_normal = 1 << 23
  smallest_normal_r0 = tf.constant(smallest_normal, dtype=tf.int32)
  smallest_normal_r2 = tf.fill([height, width], smallest_normal_r0)

  # Used to mask away the sign bit when computing the absolute value.
  low_bit_mask = ~(1 << 31)
  low_bit_mask_r0 = tf.constant(low_bit_mask, dtype=tf.int32)
  low_bit_mask_r2 = tf.fill([height, width], low_bit_mask_r0)

  iota = tf.tile(tf.expand_dims(tf.range(width, dtype=tf.int32), 0),
                 [height, 1])

  # Compare the absolute value with positive zero to handle negative zero.
  #
  # Pseudocode: input_no_zeros = abs(input) == 0 ? FLT_MIN : input
  input_r2 = tf.bitcast(inputs, tf.int32)
  abs_r2 = tf.bitwise.bitwise_and(input_r2, low_bit_mask_r2)
  if_zero_r2 = tf.equal(abs_r2, zeros)
  smallest_normal_preserving_sign_r2 = tf.bitwise.bitwise_or(
      input_r2, smallest_normal_r2)
  input_no_zeros_r2 = tf.where(
      if_zero_r2, smallest_normal_preserving_sign_r2, input_r2)

  # Discard the low-order bits and replace with iota.
  and_r2 = tf.bitwise.bitwise_and(input_no_zeros_r2, count_mask_r2)
  or_r2 = tf.bitwise.bitwise_or(and_r2, iota)
  return tf.bitcast(or_r2, tf.float32)


def create_topk_unique(inputs, k):
  height = inputs.shape[0]
  width = inputs.shape[1]
  neg_inf_r0 = tf.constant(-np.inf, dtype=tf.float32)
  ones = tf.ones([height, width], dtype=tf.float32)
  neg_inf_r2 = ones * neg_inf_r0
  inputs = tf.where(tf.is_nan(inputs), neg_inf_r2, inputs)

  tmp = inputs
  topk_r2 = tf.zeros([height, k], dtype=tf.float32)
  for i in range(k):
    kth_order_statistic = tf.reduce_max(tmp, axis=1, keepdims=True)
    k_mask = tf.tile(tf.expand_dims(tf.equal(tf.range(k), tf.fill([k], i)), 0),
                     [height, 1])
    topk_r2 = tf.where(k_mask, tf.tile(kth_order_statistic, [1, k]), topk_r2)
    ge_r2 = tf.greater_equal(inputs, tf.tile(kth_order_statistic, [1, width]))
    tmp = tf.where(ge_r2, neg_inf_r2, inputs)

  log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = next_power_of_two - 1
  mask_r0 = tf.constant(count_mask)
  mask_r2 = tf.fill([height, k], mask_r0)
  topk_r2_s32 = tf.bitcast(topk_r2, tf.int32)
  topk_indices_r2 = tf.bitwise.bitwise_and(topk_r2_s32, mask_r2)
  return topk_r2, topk_indices_r2


def top_k_with_unique(inputs, k):
  unique_inputs = create_make_unique(tf.cast(inputs, tf.float32))
  top_values, indices = create_topk_unique(unique_inputs, k)
  top_values = tf.cast(top_values, inputs.dtype)
  return top_values, indices


def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight, max_tgt):
  """Performs a single step of Beam Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape
      `[batch_size, beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search.
      An instance of `BeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.
    max_tgt: maximum prediction length.

  Returns:
    A new beam state.
  """

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished
  not_finished = tf.logical_not(previously_finished)

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  step_log_probs = tf.nn.log_softmax(logits)
  step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
  total_probs = tf.expand_dims(beam_state.log_probs, 2) + step_log_probs

  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.shape[-1].value or tf.shape(logits)[-1]
  lengths_to_add = tf.one_hot(
      indices=tf.fill([batch_size, beam_width], end_token),
      depth=vocab_size,
      on_value=np.int64(0),
      off_value=np.int64(1),
      dtype=tf.int64)
  add_mask = tf.to_int64(not_finished)
  lengths_to_add *= tf.expand_dims(add_mask, 2)
  new_prediction_lengths = (
      lengths_to_add + tf.expand_dims(prediction_lengths, 2))

  # Calculate the accumulated attention probabilities if coverage penalty is
  # enabled.
  accumulated_attention_probs = None
  attention_probs = get_attention_probs(
      next_cell_state, coverage_penalty_weight)
  if attention_probs is not None:
    attention_probs *= tf.expand_dims(
        tf.cast(not_finished, attention_probs.dtype), 2)
    accumulated_attention_probs = (
        beam_state.accumulated_attention_probs + attention_probs)

  batch_finished = tf.reduce_all(
      previously_finished, axis=1, keepdims=True)
  any_batch_finished = tf.reduce_any(batch_finished)
  batch_finished = tf.tile(tf.expand_dims(batch_finished, 2),
                           [1, beam_width, vocab_size])
  def _normalized_scores():
    return _get_scores(
        log_probs=total_probs,
        sequence_lengths=new_prediction_lengths,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        finished=batch_finished,
        accumulated_attention_probs=accumulated_attention_probs)

  # Normalize the scores of finished batches.
  scores = tf.cond(any_batch_finished, _normalized_scores, lambda: total_probs)

  time = tf.convert_to_tensor(time, name="time")
  # During the first time step we only consider the initial beam
  scores_flat = tf.reshape(scores, [batch_size, -1])

  # Pick the next beams according to the specified successors function
  next_beam_scores, word_indices = top_k_with_unique(scores_flat, beam_width)

  next_beam_scores.set_shape([batch_size, beam_width])
  word_indices.set_shape([batch_size, beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  batch_ids = tf.expand_dims(
      tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, beam_width]), 2)
  indices = tf.concat([batch_ids, tf.expand_dims(word_indices, 2)], -1)
  next_beam_probs = tf.gather_nd(
      tf.reshape(total_probs, [batch_size, -1]), indices)

  # Note: just doing the following
  #   tf.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = tf.mod(
      word_indices, vocab_size, name="next_beam_word_ids")
  next_word_ids = tf.to_int32(raw_next_word_ids)
  next_beam_ids = tf.div(word_indices, vocab_size, name="next_beam_parent_ids")

  # Append new ids to current predictions
  previously_finished = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_finished = tf.logical_or(
      previously_finished,
      tf.equal(next_word_ids, end_token),
      name="next_beam_finished")

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged.
  # 2. Beams that are now finished (EOS predicted) have their length
  #    increased by 1.
  # 3. Beams that are not yet finished have their length increased by 1.
  lengths_to_add = tf.to_int64(tf.logical_not(previously_finished))
  next_prediction_len = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_prediction_len += lengths_to_add
  next_accumulated_attention_probs = ()
  if accumulated_attention_probs is not None:
    next_accumulated_attention_probs = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=accumulated_attention_probs,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[batch_size * beam_width, -1],
        name="next_accumulated_attention_probs")
  next_pred_ids = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.pred_ids,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[batch_size * beam_width, -1],
      name="pred_ids")

  # Add next_word_ids to next_pred_ids.
  next_pred_ids = tf.transpose(next_pred_ids, [2, 0, 1])
  cur_time = tf.tile(tf.reshape(time, [1]), [max_tgt])
  time_mask = tf.equal(tf.range(max_tgt), cur_time)
  time_mask = tf.tile(tf.reshape(
      time_mask, [max_tgt, 1, 1]), [1, batch_size, beam_width])
  cur_time_ids = tf.tile(tf.reshape(next_word_ids, [1, batch_size, beam_width]),
                         [max_tgt, 1, 1])
  next_pred_ids = tf.where(time_mask, cur_time_ids, next_pred_ids)
  next_pred_ids = tf.transpose(next_pred_ids, [1, 2, 0])

  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = tf.contrib.framework.nest.map_structure(
      lambda gather_from: _maybe_tensor_gather_helper(
          gather_indices=next_beam_ids,
          gather_from=gather_from,
          batch_size=batch_size,
          range_size=beam_width,
          gather_shape=[batch_size * beam_width, -1]),
      next_cell_state)
  # pylint: enable=g-long-lambda

  next_state = BeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished,
      accumulated_attention_probs=next_accumulated_attention_probs,
      pred_ids=next_pred_ids)

  output = BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state


def get_attention_probs(next_cell_state, coverage_penalty_weight):
  """Get attention probabilities from the cell state.

  Args:
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.

  Returns:
    The attention probabilities with shape `[batch_size, beam_width, max_time]`
    if coverage penalty is enabled. Otherwise, returns None.

  Raises:
    ValueError: If no cell is attentional but coverage penalty is enabled.
  """
  if coverage_penalty_weight == 0.0:
    return None

  # Attention probabilities of each attention layer. Each with shape
  # `[batch_size, beam_width, max_time]`.
  probs_per_attn_layer = []
  if isinstance(next_cell_state, attention_utils.AttentionWrapperState):
    probs_per_attn_layer = [attention_probs_from_attn_state(next_cell_state)]
  elif isinstance(next_cell_state, tuple):
    for state in next_cell_state:
      if isinstance(state, attention_utils.AttentionWrapperState):
        probs_per_attn_layer.append(attention_probs_from_attn_state(state))

  if not probs_per_attn_layer:
    raise ValueError(
        "coverage_penalty_weight must be 0.0 if no cell is attentional.")

  if len(probs_per_attn_layer) == 1:
    attention_probs = probs_per_attn_layer[0]
  else:
    # Calculate the average attention probabilities from all attention layers.
    attention_probs = [
        tf.expand_dims(prob, -1) for prob in probs_per_attn_layer]
    attention_probs = tf.concat(attention_probs, -1)
    attention_probs = tf.reduce_mean(attention_probs, -1)

  return attention_probs


def _get_scores(log_probs, sequence_lengths, length_penalty_weight,
                coverage_penalty_weight, finished, accumulated_attention_probs):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape
      `[batch_size, beam_width, vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.
    finished: A boolean tensor of shape `[batch_size, beam_width, vocab_size]`
      that specifies which elements in the beam are finished already.
    accumulated_attention_probs: Accumulated attention probabilities up to the
      current time step, with shape `[batch_size, beam_width, max_time]` if
      coverage_penalty_weight is not 0.0.

  Returns:
    The scores normalized by the length_penalty and coverage_penalty.

  Raises:
    ValueError: accumulated_attention_probs is None when coverage penalty is
      enabled.
  """
  length_penalty_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight,
      dtype=log_probs.dtype)

  if coverage_penalty_weight == 0.0:
    return tf.where(
      finished, log_probs / length_penalty_, log_probs)

  coverage_penalty_weight = tf.convert_to_tensor(
      coverage_penalty_weight, name="coverage_penalty_weight",
      dtype=log_probs.dtype)
  if coverage_penalty_weight.shape.ndims != 0:
    raise ValueError("coverage_penalty_weight should be a scalar, "
                     "but saw shape: %s" % coverage_penalty_weight.shape)

  if accumulated_attention_probs is None:
    raise ValueError(
        "accumulated_attention_probs can be None only if coverage penalty is "
        "disabled.")

  # Add source sequence length mask before computing coverage penalty.
  accumulated_attention_probs = tf.where(
      tf.equal(accumulated_attention_probs, 0.0),
      tf.ones_like(accumulated_attention_probs),
      accumulated_attention_probs)

  # coverage penalty =
  #     sum over `max_time` {log(min(accumulated_attention_probs, 1.0))}
  coverage_penalty = tf.reduce_sum(
      tf.log(tf.minimum(accumulated_attention_probs, 1.0)), 2)
  # Apply coverage penalty to finished predictions.
  weighted_coverage_penalty = coverage_penalty * coverage_penalty_weight
  # Reshape from [batch_size, beam_width] to [batch_size, beam_width, 1]
  weighted_coverage_penalty = tf.expand_dims(
      weighted_coverage_penalty, 2)

  # Normalize the scores of finished predictions.
  return tf.where(
      finished, log_probs / length_penalty_ + weighted_coverage_penalty,
      log_probs)


def attention_probs_from_attn_state(attention_state):
  """Calculates the average attention probabilities.

  Args:
    attention_state: An instance of `AttentionWrapperState`.

  Returns:
    The attention probabilities in the given AttentionWrapperState.
    If there're multiple attention mechanisms, return the average value from
    all attention mechanisms.
  """
  # Attention probabilities over time steps, with shape
  # `[batch_size, beam_width, max_time]`.
  attention_probs = attention_state.alignments
  if isinstance(attention_probs, tuple):
    attention_probs = [
        tf.expand_dims(prob, -1) for prob in attention_probs]
    attention_probs = tf.concat(attention_probs, -1)
    attention_probs = tf.reduce_mean(attention_probs, -1)
  return attention_probs


def _length_penalty(sequence_lengths, penalty_factor, dtype):
  """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

  Returns the length penalty tensor:
  ```
  [(5+sequence_lengths)/6]**penalty_factor
  ```
  where all operations are performed element-wise.

  Args:
    sequence_lengths: `Tensor`, the sequence lengths of each hypotheses.
    penalty_factor: A scalar that weights the length penalty.
    dtype: dtype of result.

  Returns:
    If the penalty is `0`, returns the scalar `1.0`.  Otherwise returns
    the length penalty factor, a tensor with the same shape as
    `sequence_lengths`.
  """
  penalty_factor = tf.convert_to_tensor(
      penalty_factor, name="penalty_factor", dtype=dtype)
  penalty_factor.set_shape(())  # penalty should be a scalar.
  static_penalty = tf.contrib.util.constant_value(penalty_factor)
  if static_penalty is not None and static_penalty == 0:
    return 1.0
  length_penalty_const = 5.0
  mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_LEN_NORM_CONST,
                        value=length_penalty_const)
  return tf.div((length_penalty_const + tf.cast(sequence_lengths, dtype))
                **penalty_factor, (length_penalty_const + 1.)**penalty_factor)


def _mask_probs(probs, eos_token, finished):
  """Masks log probabilities.

  The result is that finished beams allocate all probability mass to eos and
  unfinished beams remain unchanged.

  Args:
    probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
    beams stay unchanged and finished beams are replaced with a tensor with all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[2]
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=probs.dtype,
      on_value=tf.convert_to_tensor(0., dtype=probs.dtype),
      off_value=probs.dtype.min)
  finished_probs = tf.tile(
      tf.reshape(finished_row, [1, 1, -1]),
      tf.concat([tf.shape(finished), [1]], 0))
  finished_mask = tf.tile(
      tf.expand_dims(finished, 2), [1, 1, vocab_size])

  return tf.where(finished_mask, finished_probs, probs)


def _maybe_tensor_gather_helper(gather_indices, gather_from, batch_size,
                                range_size, gather_shape):
  """Maybe applies _tensor_gather_helper.

  This applies _tensor_gather_helper when the gather_from dims is at least as
  big as the length of gather_shape. This is used in conjunction with nest so
  that we don't apply _tensor_gather_helper to inapplicable values like scalars.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
      or the original tensor if its dimensions are too small.
  """
  if isinstance(gather_from, tf.TensorArray):
    return gather_from
  _check_maybe(gather_from)
  if gather_from.shape.ndims >= len(gather_shape):
    return _tensor_gather_helper(
        gather_indices=gather_indices,
        gather_from=gather_from,
        batch_size=batch_size,
        range_size=range_size,
        gather_shape=gather_shape)
  else:
    return gather_from


def _tensor_gather_helper(gather_indices,
                          gather_from,
                          batch_size,
                          range_size,
                          gather_shape,
                          name=None):
  """Helper for gathering the right indices from the tensor.

  This works by reshaping gather_from to gather_shape (e.g. [-1]) and then
  gathering from that according to the gather_indices, which are offset by
  the right amounts in order to preserve the batch order.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The input batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.
    name: The tensor name for set of operations. By default this is
      'tensor_gather_helper'. The final output is named 'output'.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
  """
  with tf.name_scope(name, "tensor_gather_helper"):
    range_ = tf.expand_dims(tf.range(batch_size) * range_size, 1)
    gather_indices = tf.reshape(gather_indices + range_, [-1])
    output = tf.gather(
        tf.reshape(gather_from, gather_shape), gather_indices)
    final_shape = tf.shape(gather_from)[:1 + len(gather_shape)]
    final_static_shape = (
        tf.TensorShape([batch_size]).concatenate(
            gather_from.shape[1:1 + len(gather_shape)]))
    output = tf.reshape(output, final_shape, name="output")
    output.set_shape(final_static_shape)
    return output
