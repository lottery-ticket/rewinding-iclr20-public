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

"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time
from distutils import version
import tensorflow as tf


def bfloat16_var_getter(getter, *args, **kwargs):
  """A custom getter function for bfloat16 variables.

  Variables maintain storage in float32.

  Args:
    getter: custom getter
    *args: arguments
    **kwargs: keyword arguments
  Returns:
    variables with the correct dtype.
  Raises:
    KeyError: if "dtype" is not provided as a kwarg.
  """
  requested_dtype = kwargs["dtype"]
  if requested_dtype == tf.bfloat16:
    kwargs["dtype"] = tf.float32
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = "1.3.0"
  # LINT
  if (version.LooseVersion(tf.__version__) <
      version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))


def maybe_parse_standard_hparams(hparams, hparams_path):
  """Override hparams values with existing standard hparams config."""
  if hparams_path and tf.gfile.Exists(hparams_path):
    print_out("# Loading standard hparams from %s" % hparams_path)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_path, "rb")) as f:
      hparams.parse_json(f.read())
  return hparams


def format_text(words):
  """Convert a sequence words into sentence."""
  if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
    words = [words]
  return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
  """Convert a sequence of bpe words into sentence."""
  words = []
  word = b""
  if isinstance(symbols, str):
    symbols = symbols.encode()
  delimiter_len = len(delimiter)
  for symbol in symbols:
    if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
      word += symbol[:-delimiter_len]
    else:  # end of a word
      word += symbol
      words.append(word)
      word = b""
  return b" ".join(words)


def format_spm_text(symbols):
  """Decode a text in SPM (https://github.com/google/sentencepiece) format."""
  return u"".join(format_text(symbols).decode("utf-8").split()).replace(
      u"\u2581", u" ").strip().encode("utf-8")
