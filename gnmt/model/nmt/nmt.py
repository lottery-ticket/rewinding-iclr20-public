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

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from tensorflow.contrib.training.python.training import evaluation
from mlperf_compliance import mlperf_log
import distributed_iterator_utils
import estimator
from utils import misc_utils as utils
from utils import vocab_utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument(
      "--num_units", type=int, default=1024, help="Network size.")
  parser.add_argument(
      "--num_layers", type=int, default=4, help="Network depth.")
  parser.add_argument("--num_encoder_layers", type=int, default=None,
                      help="Encoder depth, equal to num_layers if None.")
  parser.add_argument("--num_decoder_layers", type=int, default=None,
                      help="Decoder depth, equal to num_layers if None.")
  parser.add_argument(
      "--encoder_type",
      type=str,
      default="gnmt",
      help="""\
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.\
      """)
  parser.add_argument(
      "--residual",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Whether to add residual connections.")
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      default=True,
                      help="Whether to use time-major mode for dynamic RNN.")
  parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                      help="Number of partitions for embedding vars.")

  # attention mechanisms
  parser.add_argument(
      "--attention",
      type=str,
      default="normed_bahdanau",
      help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
  parser.add_argument(
      "--attention_architecture",
      type=str,
      default="gnmt_v2",
      help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
  parser.add_argument(
      "--output_attention", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Only used in standard attention_architecture. Whether use attention as
      the cell output at each timestep.
      .\
      """)
  parser.add_argument(
      "--pass_hidden_state", type="bool", nargs="?", const=True,
      default=True,
      help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

  # optimizer
  parser.add_argument(
      "--optimizer", type=str, default="adam", help="sgd | adam")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument("--warmup_steps", type=int, default=0,
                      help="How many steps we inverse-decay learning.")
  parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
  parser.add_argument(
      "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

  parser.add_argument(
      "--max_train_epochs", type=int, default=10,
      help="Maximum number of training epochs.")
  parser.add_argument("--num_examples_per_epoch", type=int, default=3534981,
                      help="Number of examples in one epoch")
  parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                      const=True,
                      default=True,
                      help=("Whether try colocating gradients with "
                            "corresponding op"))
  parser.add_argument("--label_smoothing", type=float, default=0.1,
                      help=("If nonzero, smooth the labels towards "
                            "1/num_classes."))

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data
  parser.add_argument(
      "--src", type=str, default="en", help="Source suffix, e.g., en.")
  parser.add_argument(
      "--tgt", type=str, default="de", help="Target suffix, e.g., de.")
  parser.add_argument(
      "--data_dir", type=str, default="", help="Training/eval data directory.")

  parser.add_argument(
      "--train_prefix",
      type=str,
      default="train.tok.clean.bpe.32000",
      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
      "--test_prefix",
      type=str,
      default="newstest20{year}.tok.bpe.32000.padded",
      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument(
    '--test_year',
    type=str,
    default='14',
    help='TEST YEAR')
  parser.add_argument(
    '--test_force',
    type=bool,
    default=False,
    help='FORCE TEST')

  parser.add_argument(
      "--out_dir", type=str, default=None, help="Store log/model files.")

  # Vocab
  parser.add_argument(
      "--vocab_prefix",
      type=str,
      default="vocab.bpe.32000",
      help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)

  parser.add_argument(
      "--embed_prefix",
      type=str,
      default=None,
      help="""\
      Pretrained embedding prefix, expect files with src/tgt suffixes.
      The embedding files should be Glove formatted txt files.\
      """)
  parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
  parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")
  parser.add_argument(
      "--share_vocab",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)
  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths
  parser.add_argument(
      "--src_max_len",
      type=int,
      default=48,
      help="Max length of src sequences during training.")
  parser.add_argument(
      "--tgt_max_len",
      type=int,
      default=48,
      help="Max length of tgt sequences during training.")
  parser.add_argument("--src_max_len_infer", type=int, default=150,
                      help="Max length of src sequences during inference.")
  parser.add_argument("--tgt_max_len_infer", type=int, default=150,
                      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change)
  parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | gru | layer_norm_lstm | nas")
  parser.add_argument("--forget_bias", type=float, default=0.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=5,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument(
      "--num_buckets",
      type=int,
      default=5,
      help="Put data into similar-length buckets.")
  parser.add_argument(
      "--choose_buckets",
      type=int,
      default=None,
      help="Choose from this number of length buckets per training step.")
  parser.add_argument("--num_sampled_softmax", type=int, default=0,
                      help=("Use sampled_softmax_loss if > 0."
                            "Otherwise, use full softmax loss."))

  # SPM
  parser.add_argument("--subword_option", type=str, default="bpe",
                      choices=["", "bpe", "spm"],
                      help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

  # Experimental encoding feature.
  parser.add_argument("--use_char_encode", type="bool", default=False,
                      help="""\
                      Whether to split each word or bpe into character, and then
                      generate the word-level representation from the character
                      reprentation.
                      """)

  # Misc
  parser.add_argument(
      "--num_shards", type=int,
      default=8, help="Number of shards (TPU cores).")
  parser.add_argument(
      "--num_shards_per_host", type=int,
      default=8, help="Number of shards (TPU cores) per host.")
  parser.add_argument(
      "--num_gpus", type=int, default=4, help="Number of gpus in each worker.")
  parser.add_argument(
      "--num_tpu_workers",
      type=int,
      default=None,
      help="Number of TPU workers; if set, uses the distributed-sync pipeline.")
  parser.add_argument(
      "--log_device_placement",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Debug GPU allocation.")
  parser.add_argument("--scope", type=str, default=None,
                      help="scope to put variables under")
  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument(
      "--random_seed",
      type=int,
      default=1,
      help="Random seed (>0, set a specific seed).")
  parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                      const=True, default=False,
                      help="Override loaded hparams with values specified")
  parser.add_argument("--language_model", type="bool", nargs="?",
                      const=True, default=False,
                      help="True to train a language model, ignoring encoder")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument(
      "--infer_batch_size",
      type=int,
      default=512,
      help="Batch size for inference mode.")
  parser.add_argument(
      "--examples_to_infer",
      type=int,
      default=3003,
      help="Number of examples to infer.")
  parser.add_argument("--detokenizer_file", type=str,
                      default="mosesdecoder/scripts/tokenizer/detokenizer.perl",
                      help=("""Detokenizer script file."""))
  parser.add_argument("--use_borg", type=bool, default=False)
  parser.add_argument("--target_bleu", type=float, default=21.8,
                      help="Target accuracy.")

  # Advanced inference arguments
  parser.add_argument("--infer_mode", type=str, default="beam_search",
                      choices=["greedy", "sample", "beam_search"],
                      help="Which type of decoder to use during inference.")
  parser.add_argument("--beam_width", type=int, default=5,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument(
      "--length_penalty_weight",
      type=float,
      default=0.6,
      help="Length penalty for beam search.")
  parser.add_argument(
      "--coverage_penalty_weight",
      type=float,
      default=0.1,
      help="Coverage penalty for beam search.")
  parser.add_argument("--sampling_temperature", type=float,
                      default=0.0,
                      help=("""\
      Softmax sampling temperature for inference decoding, 0.0 means greedy
      decoding. This option is ignored when using beam search.\
      """))

  # Job info
  parser.add_argument("--jobid", type=int, default=0,
                      help="Task id of the worker.")

  # TPU
  parser.add_argument("--use_tpu", type=bool, default=True)
  parser.add_argument("--use_tpu_low_level_api", type=bool, default=True)
  parser.add_argument("--master", type=str, default="",
                      help=("Address of the master. Either --master or "
                            "--tpu_name must be specified."))
  parser.add_argument("--tpu_name", type=str, default=None,
                      help=("Name of the TPU for Cluster Resolvers. Either "
                            "--tpu_name or --master must be specified."))
  parser.add_argument("--use_dynamic_rnn", type=bool, default=False)
  parser.add_argument("--use_synthetic_data", type=bool, default=False)
  parser.add_argument(
      "--mode", type=str, default="train_and_eval",
      choices=["train", "train_and_eval", "infer"])
  parser.add_argument("--activation_dtype", type=str, default="float32",
                      choices=["float32", "bfloat16"])
  parser.add_argument("--use_async_checkpoint", type=bool, default=True)


  parser.add_argument("--lottery_results_dir")
  parser.add_argument("--lottery_checkpoint_iters")
  parser.add_argument("--lottery_prune_at")
  parser.add_argument("--lottery_pruning_method")
  parser.add_argument("--lottery_reset_to")
  parser.add_argument("--lottery_reset_global_step_to")
  parser.add_argument("--lottery_force_learning_rate", type=float)
  parser.add_argument("--model_dir")



def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.data_dir + flags.train_prefix,
      test_prefix=flags.data_dir + flags.test_prefix.format(year=flags.test_year),
      test_year=flags.test_year,
      test_force=flags.test_force,
      vocab_prefix=flags.data_dir + flags.vocab_prefix,
      embed_prefix=flags.embed_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units,
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
      dropout=flags.dropout,
      unit_type=flags.unit_type,
      encoder_type=flags.encoder_type,
      residual=flags.residual,
      time_major=flags.time_major,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Attention mechanisms
      attention=flags.attention,
      attention_architecture=flags.attention_architecture,
      output_attention=flags.output_attention,
      pass_hidden_state=flags.pass_hidden_state,

      # Train
      optimizer=flags.optimizer,
      max_train_epochs=flags.max_train_epochs,
      num_examples_per_epoch=flags.num_examples_per_epoch,
      batch_size=flags.batch_size,
      num_train_steps=int(flags.num_examples_per_epoch / flags.batch_size * 10),
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      label_smoothing=flags.label_smoothing,
      warmup_steps=flags.warmup_steps,
      warmup_scheme=flags.warmup_scheme,
      decay_scheme=flags.decay_scheme,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
      num_sampled_softmax=flags.num_sampled_softmax,

      # Data constraints
      num_buckets=flags.num_buckets,
      choose_buckets=flags.choose_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      examples_to_infer=flags.examples_to_infer,
      detokenizer_file=flags.data_dir + flags.detokenizer_file,
      use_borg=flags.use_borg,
      target_bleu=flags.target_bleu,

      # Advanced inference arguments
      infer_mode=flags.infer_mode,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,
      coverage_penalty_weight=flags.coverage_penalty_weight,
      sampling_temperature=flags.sampling_temperature,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,
      use_char_encode=flags.use_char_encode,

      # Misc
      forget_bias=flags.forget_bias,
      num_shards=flags.num_shards,
      num_shards_per_host=flags.num_shards_per_host,
      num_gpus=flags.num_gpus,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      share_vocab=flags.share_vocab,
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      language_model=flags.language_model,

      # TPU
      use_tpu=flags.use_tpu,
      use_tpu_low_level_api=flags.use_tpu_low_level_api,
      master=flags.master,
      tpu_name=flags.tpu_name,
      use_dynamic_rnn=flags.use_dynamic_rnn,
      use_synthetic_data=flags.use_synthetic_data,
      mode=flags.mode,
      activation_dtype=flags.activation_dtype,
      use_async_checkpoint=flags.use_async_checkpoint,

    lottery_results_dir=flags.lottery_results_dir,
    lottery_checkpoint_iters=flags.lottery_checkpoint_iters,
    lottery_prune_at=flags.lottery_prune_at,
    lottery_pruning_method=flags.lottery_pruning_method,
    lottery_reset_to=flags.lottery_reset_to,
    lottery_reset_global_step_to=flags.lottery_reset_global_step_to,
    lottery_force_learning_rate=flags.lottery_force_learning_rate,
    model_dir=flags.model_dir,

  )


def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)


def extend_hparams(hparams):
  """Add new arguments to hparams."""
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
    raise ValueError("For bi, num_encoder_layers %d should be even" %
                     hparams.num_encoder_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_encoder_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_encoder_layers %d should be >= 2" %
                     hparams.num_encoder_layers)
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")
  if hparams.infer_mode == "beam_search" and hparams.beam_width <= 0:
    raise ValueError("beam_width must greater than 0 when using beam_search"
                     "decoder.")
  if hparams.infer_mode == "sample" and hparams.sampling_temperature <= 0.0:
    raise ValueError("sampling_temperature must greater than 0.0 when using"
                     "sample decoder.")

  # Different number of encoder / decoder layers
  assert hparams.num_encoder_layers and hparams.num_decoder_layers
  if hparams.num_encoder_layers != hparams.num_decoder_layers:
    hparams.pass_hidden_state = False
    utils.print_out("Num encoder layer %d is different from num decoder layer"
                    " %d, so set pass_hidden_state to False" % (
                        hparams.num_encoder_layers,
                        hparams.num_decoder_layers))

  # Set residual layers
  num_encoder_residual_layers = 0
  num_decoder_residual_layers = 0
  if hparams.residual:
    if hparams.num_encoder_layers > 1:
      num_encoder_residual_layers = hparams.num_encoder_layers - 1
    if hparams.num_decoder_layers > 1:
      num_decoder_residual_layers = hparams.num_decoder_layers - 1

    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_encoder_residual_layers = hparams.num_encoder_layers - 2

      # Compatible for GNMT models
      if hparams.num_encoder_layers == hparams.num_decoder_layers:
        num_decoder_residual_layers = num_encoder_residual_layers
  _add_argument(hparams, "num_encoder_residual_layers",
                num_encoder_residual_layers)
  _add_argument(hparams, "num_decoder_residual_layers",
                num_decoder_residual_layers)

  # Language modeling
  if hparams.language_model:
    hparams.attention = ""
    hparams.attention_architecture = ""
    hparams.pass_hidden_state = False
    hparams.share_vocab = True
    hparams.src = hparams.tgt
    utils.print_out("For language modeling, we turn off attention and "
                    "pass_hidden_state; turn on share_vocab; set src to tgt.")

  ## Vocab
  # Get vocab file names first
  if hparams.vocab_prefix:
    src_vocab_file = hparams.vocab_prefix + "." + hparams.src
    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
  else:
    raise ValueError("hparams.vocab_prefix must be provided.")

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  mlperf_log.gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE,
                        value={"src": src_vocab_size, "tgt": tgt_vocab_size})
  _add_argument(hparams, "src_vocab_size", src_vocab_size)
  _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
  _add_argument(hparams, "src_vocab_file", src_vocab_file)
  _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

  # Num embedding partitions
  _add_argument(
      hparams, "num_enc_emb_partitions", hparams.num_embeddings_partitions)
  _add_argument(
      hparams, "num_dec_emb_partitions", hparams.num_embeddings_partitions)

  # Pretrained Embeddings
  _add_argument(hparams, "src_embed_file", "")
  _add_argument(hparams, "tgt_embed_file", "")
  if hparams.embed_prefix:
    src_embed_file = hparams.embed_prefix + "." + hparams.src
    tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

    if tf.gfile.Exists(src_embed_file):
      utils.print_out("  src_embed_file %s exist" % src_embed_file)
      hparams.src_embed_file = src_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_enc_emb_partitions to 1")
      hparams.num_enc_emb_partitions = 1
    else:
      utils.print_out("  src_embed_file %s doesn't exist" % src_embed_file)

    if tf.gfile.Exists(tgt_embed_file):
      utils.print_out("  tgt_embed_file %s exist" % tgt_embed_file)
      hparams.tgt_embed_file = tgt_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_dec_emb_partitions to 1")
      hparams.num_dec_emb_partitions = 1
    else:
      utils.print_out("  tgt_embed_file %s doesn't exist" % tgt_embed_file)

  return hparams


def create_or_load_hparams(default_hparams, hparams_path):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.maybe_parse_standard_hparams(default_hparams, hparams_path)
  hparams = extend_hparams(hparams)
  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def run_main(flags, default_hparams, estimator_fn):
  """Run main."""
  # Job
  jobid = flags.jobid
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)
    tf.set_random_seed(random_seed)

  # Model output directory
  out_dir = flags.out_dir
  if out_dir and not tf.gfile.Exists(out_dir):
    utils.print_out("# Creating output directory %s ..." % out_dir)
    tf.gfile.MakeDirs(out_dir)

  # Load hparams.
  hparams = create_or_load_hparams(default_hparams, flags.hparams_path)

  # Train or Evaluation
  return estimator_fn(hparams)


def main(unused_argv):
  # pylint: disable=g-long-lambda
  if FLAGS.mode == "train":
    print("Running training mode.")
    mlperf_log.gnmt_print(key=mlperf_log.RUN_START)
    default_hparams = create_hparams(FLAGS)
    if FLAGS.num_tpu_workers:
      _ = run_main(
          FLAGS, default_hparams,
          lambda hparams: distributed_iterator_utils.train_fn(
              hparams, FLAGS.num_tpu_workers))
    else:
      _ = run_main(FLAGS, default_hparams, estimator.train_fn)
  elif FLAGS.mode == "train_and_eval":
    print("Running training and evaluation mode.")
    default_hparams = create_hparams(FLAGS)
    if FLAGS.num_tpu_workers:
      if FLAGS.use_tpu_low_level_api:
        _ = run_main(
            FLAGS, default_hparams,
            lambda hparams: distributed_iterator_utils.train_and_eval_with_low_level_api(
                hparams, FLAGS.num_tpu_workers))
      else:
        _ = run_main(
            FLAGS, default_hparams,
            lambda hparams: distributed_iterator_utils.train_and_eval_fn(
                hparams, FLAGS.num_tpu_workers))
    else:
      if FLAGS.use_tpu_low_level_api:
        _ = run_main(FLAGS, default_hparams,
                     estimator.train_and_eval_with_low_level_api)
      else:
        _ = run_main(FLAGS, default_hparams, estimator.train_and_eval_fn)
    mlperf_log.gnmt_print(key=mlperf_log.RUN_FINAL)
  else:
    print("Running inference mode.")
    default_hparams = create_hparams(FLAGS)

    if (not default_hparams.test_force) and tf.gfile.Exists(os.path.join(default_hparams.out_dir, "eval_{}".format(default_hparams.test_year), 'bleu')):
      return

    current_epoch = 0
    mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET,
                          value=default_hparams.target_bleu)
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(FLAGS.out_dir):
      # Terminate eval job once target score is reached
      current_step = int(os.path.basename(ckpt).split("-")[1])
      if current_step == 0:
        current_epoch = 0
      else:
        current_epoch += 1

      tf.logging.info("Starting to evaluate...%s", ckpt)
      try:
        mlperf_log.gnmt_print(
            key=mlperf_log.TRAIN_CHECKPOINT, value=("Under " + ckpt))
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_START)
        score = run_main(FLAGS, default_hparams, estimator.eval_fn)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                              value={"value": score, "epoch": current_epoch})
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP, value=current_epoch)
        if score > FLAGS.target_bleu:
          tf.logging.info(
              "Evaluation finished after training step %d" % current_step)
          mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": True})
          break
        # Terminate eval job when final checkpoint is reached
        max_steps = default_hparams.num_train_steps
        if current_step >= max_steps:
          tf.logging.info(
              "Evaluation finished but failed to reach target score.")
          mlperf_log.gnmt_print(mlperf_log.RUN_STOP, {"success": False})
          break

      except tf.errors.NotFoundError:
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping checkpoint" % ckpt)
    mlperf_log.gnmt_print(key=mlperf_log.RUN_FINAL)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
