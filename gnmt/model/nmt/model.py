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
"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import tensorflow as tf

from mlperf_compliance import mlperf_log
import beam_search_decoder
import decoder
import model_helper
from utils import misc_utils as utils

from lottery import lottery

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self, hparams, mode, features, scope=None, extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      features: a dict of input features.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    # Set params
    self._set_params_initializer(hparams, mode, features, scope, extra_args)

    # Train graph
    res = self.build_graph(hparams, scope=scope)
    self._set_train_or_infer(res, hparams)


  def _emb_lookup(self, weight, index, is_decoder=False):
    return tf.cast(
        tf.reshape(
            tf.gather(weight, tf.reshape(index, [-1])),
            [index.shape[0], index.shape[1], -1]), self.dtype)

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              features,
                              scope,
                              extra_args=None):
    """Set various params for self and initialize."""
    self.mode = mode
    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.features = features
    self.time_major = hparams.time_major
    if self.time_major:
      mlperf_log.gnmt_print(key=mlperf_log.INPUT_ORDER, value="time_major")
    else:
      mlperf_log.gnmt_print(key=mlperf_log.INPUT_ORDER, value="batch_major")

    if hparams.use_char_encode:
      assert (not self.time_major), ("Can't use time major for"
                                     " char-level inputs.")

    self.dtype = tf.as_dtype(hparams.activation_dtype)

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num units
    mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_HIDDEN_SIZE,
                          value=hparams.num_units)
    self.num_units = hparams.num_units
    self.eos_id = hparams.tgt_eos_id
    self.label_smoothing = hparams.label_smoothing

    # Set num layers
    mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_NUM_LAYERS,
                          value={"encoder": hparams.num_encoder_layers,
                                 "decoder": hparams.num_decoder_layers})
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Set num residual layers
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Batch size
    self.batch_size = tf.size(self.features["source_sequence_length"])

    # Global step
    # Use get_global_step instead of user-defied global steps. Otherwise the
    # num_train_steps in TPUEstimator.train has no effect (will train forever).
    # TPUestimator only check if tf.train.get_global_step() < num_train_steps
    self.global_step = tf.train.get_or_create_global_step()

    # Initializer
    mlperf_log.gnmt_print(key=mlperf_log.RUN_SET_RANDOM_SEED,
                          value=hparams.random_seed)
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.encoder_emb_lookup_fn = (
        self._emb_lookup if self.mode == tf.contrib.learn.ModeKeys.TRAIN else
        tf.nn.embedding_lookup)
    self.init_embeddings(hparams, scope, self.dtype)

  def _set_train_or_infer(self, res, hparams):
    """Set up training."""
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.predicted_ids = res[1]

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      loss = res[0]
      self.loss = loss

      mlperf_log.gnmt_print(key=mlperf_log.OPT_LR, value=hparams.learning_rate)

      if hparams.lottery_force_learning_rate is not None:
        self.learning_rate = lottery.get_lr_tensor(hparams.values())
      else:
        self.learning_rate = tf.constant(hparams.learning_rate)
        # warm-up
        self.learning_rate = self._get_learning_rate_warmup(hparams)
        # decay
        self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      mlperf_log.gnmt_print(key=mlperf_log.OPT_NAME, value=hparams.optimizer)
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=0.9)
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=0.999)
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=1e-8)
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      if hparams.use_tpu:
        opt = tf.contrib.tpu.CrossShardOptimizer(opt)
      # Gradients

      gradients = tf.gradients(
          loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
      clipped_grads, grad_norm = model_helper.gradient_clip(gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.update = opt.apply_gradients(zip(clipped_grads, params), global_step=self.global_step)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_decay_info(self, hparams):
    """Return decay info based on decay_scheme."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    return start_decay_step, decay_steps, decay_factor

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme, start_decay_step,
                                         decay_steps, decay_factor))

    decay_lr = tf.train.exponential_decay(
      self.learning_rate,
      (self.global_step - start_decay_step),
      decay_steps, decay_factor, staircase=True,
    )
    thresh_lr = tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: decay_lr,
    )
    thresh_lr = tf.cond(
      self.global_step >= hparams.num_train_steps,
      lambda: self.learning_rate * 0.125,
      lambda: thresh_lr,
    )

    return thresh_lr

  def init_embeddings(self, hparams, scope, dtype):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.num_units,
            tgt_embed_size=self.num_units,
            num_enc_partitions=hparams.num_enc_emb_partitions,
            num_dec_partitions=hparams.num_dec_emb_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            use_char_encode=hparams.use_char_encode,
            scope=scope,
        ))

  def _build_model(self, hparams):
    """Builds a sequence-to-sequence model.

    Args:
      hparams: Hyperparameter configurations.

    Returns:
      For infrence, A tuple of the form
      (logits, decoder_cell_outputs, predicted_ids),
      where:
        logits: logits output of the decoder.
        decoder_cell_outputs: the output of decoder.
        predicted_ids: predicted ids from beam search.
      For training, returns the final loss

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    # Encoder
    if hparams.language_model:  # no encoder for language modeling
      utils.print_out("  language modeling: no encoder")
      self.encoder_outputs = None
      encoder_state = None
    else:
      self.encoder_outputs, encoder_state = self._build_encoder(hparams)

    ## Decoder
    return self._build_decoder(self.encoder_outputs, encoder_state, hparams)

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, predicted_ids) for infererence and
      (loss, None) for training.
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols]
        loss: float32 scalar
        predicted_ids: predicted ids from beam search.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):

        output_layer = tf.get_variable(
            lottery.weight_name_of_base_name("output_projection"), [self.num_units, self.tgt_vocab_size])
        output_layer_mask = tf.get_variable(
            lottery.mask_name_of_base_name("output_projection"), [self.num_units, self.tgt_vocab_size],
          trainable=False, initializer=tf.initializers.ones()
        )
        self.output_layer = tf.math.multiply(output_layer, output_layer_mask)

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      if hparams.activation_dtype == "bfloat16":
        tf.get_variable_scope().set_custom_getter(
            utils.bfloat16_var_getter if hparams.activation_dtype == "bfloat16"
            else None)
        logits_or_loss, decoder_cell_outputs, predicted_ids = self._build_model(
            hparams)
        if decoder_cell_outputs is not None:
          decoder_cell_outputs = tf.cast(decoder_cell_outputs, tf.float32)
      else:
        logits_or_loss, decoder_cell_outputs, predicted_ids = self._build_model(
            hparams)

    return logits_or_loss, predicted_ids

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self,
                          hparams,
                          num_layers,
                          num_residual_layers,
                          fast_reverse=False,
                          reverse=False):
    """Build a multi-layer RNN cell that can be used by encoder."""
    mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_DROPOUT,
                          value=hparams.dropout)
    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn,
        global_step=self.global_step,
        fast_reverse=fast_reverse,
        seq_len=self.features["source_sequence_length"] if reverse else None)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(
          tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _compute_loss(self, theta, _, inputs):
    logits = tf.cast(
        tf.matmul(tf.slice(inputs, [0, 0], [512, self.num_units]), theta),
        tf.float32)
    target = tf.cast(
        tf.reshape(tf.slice(inputs, [0, self.num_units], [512, 1]), [-1]),
        tf.int32)
    mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_SMOOTHING,
                          value=self.label_smoothing)
    mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                          value="Cross Entropy with label smoothing")
    crossent = tf.losses.softmax_cross_entropy(
        tf.one_hot(target, self.tgt_vocab_size, dtype=logits.dtype),
        logits,
        label_smoothing=self.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    crossent = tf.where(target == self.eos_id, tf.zeros_like(crossent),
                        crossent)
    return tf.reshape(crossent, [-1]), []

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      For inference, A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
      For training, returns the final loss
    """
    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          self.features["source_sequence_length"])

      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = self.features["target_input"]
        if self.time_major:
          target_input = tf.transpose(target_input)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
          decoder_emb_inp = self._emb_lookup(
              self.embedding_decoder, target_input, is_decoder=True)
        else:
          decoder_emb_inp = tf.cast(
              tf.nn.embedding_lookup(self.embedding_decoder, target_input),
              self.dtype)

        if hparams.use_dynamic_rnn:
          final_rnn_outputs, _ = tf.nn.dynamic_rnn(
              cell,
              decoder_emb_inp,
              sequence_length=self.features["target_sequence_length"],
              initial_state=decoder_initial_state,
              dtype=self.dtype,
              scope=decoder_scope,
              time_major=self.time_major)
        else:
          final_rnn_outputs, _ = tf.contrib.recurrent.functional_rnn(
              cell,
              decoder_emb_inp,
              sequence_length=self.features["target_sequence_length"],
              initial_state=decoder_initial_state,
              dtype=self.dtype,
              scope=decoder_scope,
              time_major=self.time_major,
              use_tpu=hparams.use_tpu)

        # 512 batch dimension yields best tpu efficiency.
        factor = tf.maximum(1, 512 // self.batch_size)
        factored_batch = self.batch_size * factor
        input1 = tf.reshape(final_rnn_outputs,
                            [-1, factored_batch, self.num_units])
        input2 = tf.reshape(
            tf.transpose(self.features["target_output"]),
            [-1, factored_batch, 1])
        max_length = tf.reduce_max(self.features["target_sequence_length"])
        max_length = tf.where(
            tf.equal(max_length % factor, 0), max_length // factor,
            max_length // factor + 1)
        inputs = tf.concat(
            [tf.cast(input1, tf.float32),
             tf.cast(input2, tf.float32)], 2)

        loss, _ = tf.contrib.recurrent.Recurrent(
            theta=self.output_layer,
            state0=tf.zeros([512], tf.float32),
            inputs=inputs,
            cell_fn=self._compute_loss,
            max_input_length=max_length,
            use_tpu=True)

        return tf.reduce_sum(loss), None, None

      ## Inference
      else:
        assert hparams.infer_mode == "beam_search"
        start_tokens = tf.fill([self.batch_size], hparams.tgt_sos_id)
        end_token = hparams.tgt_eos_id
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        coverage_penalty_weight = hparams.coverage_penalty_weight

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(
            hparams, self.features["source_sequence_length"])

        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_BEAM_SIZE,
                              value=beam_width)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_MAX_SEQ_LEN,
                              value=maximum_iterations)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_LEN_NORM_FACTOR,
                              value=length_penalty_weight)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_COV_PENALTY_FACTOR,
                              value=coverage_penalty_weight)
        my_decoder = beam_search_decoder.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=self.output_layer,
            max_tgt=maximum_iterations,
            length_penalty_weight=length_penalty_weight,
            coverage_penalty_weight=coverage_penalty_weight,
            dtype=self.dtype)

        # Dynamic decoding
        predicted_ids = decoder.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

    return logits, decoder_cell_outputs, predicted_ids

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder and the initial state of
      the decoder RNN.
    """
    pass


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder_from_sequence(self, hparams, sequence, sequence_length):
    """Build an encoder from a sequence.

    Args:
      hparams: hyperparameters.
      sequence: tensor with input sequence data.
      sequence_length: tensor with length of the input sequence.

    Returns:
      encoder_outputs: RNN encoder outputs.
      encoder_state: RNN encoder state.

    Raises:
      ValueError: if encoder_type is neither "uni" nor "bi".
    """
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    if self.time_major:
      sequence = tf.transpose(sequence)

    with tf.variable_scope("encoder"):
      self.encoder_emb_inp = tf.cast(
          self.encoder_emb_lookup_fn(self.embedding_encoder, sequence),
          self.dtype)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(hparams, num_layers,
                                        num_residual_layers)

        if hparams.use_dynamic_rnn:
          encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
              cell,
              self.encoder_emb_inp,
              dtype=self.dtype,
              sequence_length=sequence_length,
              time_major=self.time_major,
              swap_memory=True)
        else:
          encoder_outputs, encoder_state = tf.contrib.recurrent.functional_rnn(
              cell,
              self.encoder_emb_inp,
              dtype=self.dtype,
              sequence_length=sequence_length,
              time_major=self.time_major,
              use_tpu=hparams.use_tpu)

      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=self.encoder_emb_inp,
                sequence_length=sequence_length,
                dtype=self.dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]

    return encoder_outputs, encoder_state

  def _build_encoder(self, hparams):
    """Build encoder from source."""
    utils.print_out("# Build a basic encoder")
    return self._build_encoder_from_sequence(
        hparams, self.features["source"],
        self.features["source_sequence_length"])

  def _build_bidirectional_rnn(self, inputs, sequence_length, dtype, hparams,
                               num_bi_layers, num_bi_residual_layers):
    """Create and call biddirectional RNN cells."""

    # num_residual_layers: Number of residual layers from top to bottom. For
    # example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2
    # RNN layers in each RNN cell will be wrapped with `ResidualWrapper`.

    fast_reverse = not hparams.use_dynamic_rnn

    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams, num_bi_layers,
                                       num_bi_residual_layers,
                                       fast_reverse)
    bw_cell = self._build_encoder_cell(hparams, num_bi_layers,
                                       num_bi_residual_layers,
                                       fast_reverse, True)

    if hparams.use_dynamic_rnn:
      bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
          fw_cell,
          bw_cell,
          inputs,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
    else:
      bi_outputs, bi_state = tf.contrib.recurrent.bidirectional_functional_rnn(
          fw_cell,
          bw_cell,
          inputs,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          use_tpu=hparams.use_tpu,
          fast_reverse=True)
      if self.mode == tf.contrib.learn.ModeKeys.INFER:
        # Remove current sequence length in cell state, which is used for fast
        # reverse.
        bi_state = tuple(s[0] for s in bi_state)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=self.num_decoder_layers,
        num_residual_layers=self.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn,
        global_step=self.global_step,
    )

    if hparams.language_model:
      encoder_state = cell.zero_state(self.batch_size, self.dtype)
    elif not hparams.pass_hidden_state:
      raise ValueError("For non-attentional model, "
                       "pass_hidden_state needs to be set to True")

    # For beam search, we need to replicate encoder infos beam_width times
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        hparams.infer_mode == "beam_search"):
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
