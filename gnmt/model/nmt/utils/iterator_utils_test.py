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
"""Tests for iterator_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from utils import iterator_utils


class IteratorUtilsTest(tf.test.TestCase):

  def testGetIterator(self):
    tf.set_random_seed(1)
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["f e a g", "c c a", "d", "c a"]))
    tgt_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["c c", "a b", "", "b c"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        num_buckets=5,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    dataset = iterator_utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=src_max_len,
        reshuffle_each_iteration=False)
    table_initializer = tf.tables_initializer()
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer)
      features = sess.run(get_next)
      self.assertAllEqual(
          [[2, 0, 3],   # c a eos -- eos is padding
           [2, 2, 0]],  # c c a
          features["source"])
      self.assertAllEqual([2, 3], features["source_sequence_length"])
      self.assertAllEqual(
          [[4, 1, 2],   # sos b c
           [4, 0, 1]],  # sos a b
          features["target_input"])
      self.assertAllEqual(
          [[1, 2, 3],   # b c eos
           [0, 1, 3]],  # a b eos
          features["target_output"])
      self.assertAllEqual([3, 3], features["target_sequence_length"])

  def testGetIteratorWithShard(self):
    tf.set_random_seed(1)
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["c c a", "f e a g", "d", "c a"]))
    tgt_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["a b", "c c", "", "b c"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        num_buckets=5,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    dataset = iterator_utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=src_max_len,
        num_shards=2,
        shard_index=1,
        reshuffle_each_iteration=False)
    table_initializer = tf.tables_initializer()
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer)
      features = sess.run(get_next)
      self.assertAllEqual(
          [[2, 0, 3],     # c a eos -- eos is padding
           [-1, -1, 0]],  # "f" == unknown, "e" == unknown, a
          features["source"])
      self.assertAllEqual([2, 3], features["source_sequence_length"])
      self.assertAllEqual(
          [[4, 1, 2],   # sos b c
           [4, 2, 2]],  # sos c c
          features["target_input"])
      self.assertAllEqual(
          [[1, 2, 3],   # b c eos
           [2, 2, 3]],  # c c eos
          features["target_output"])
      self.assertAllEqual([3, 3], features["target_sequence_length"])

  def testGetIteratorWithSkipCount(self):
    tf.set_random_seed(1)
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["c a", "c c a", "d", "f e a g"]))
    tgt_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["b c", "a b", "", "c c"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        num_buckets=5,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    skip_count = tf.placeholder(shape=(), dtype=tf.int64)
    dataset = iterator_utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=src_max_len,
        skip_count=skip_count,
        reshuffle_each_iteration=False)
    table_initializer = tf.tables_initializer()
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer, feed_dict={skip_count: 1})
      features = sess.run(get_next)

      self.assertAllEqual(
          [[-1, -1, 0],  # "f" == unknown, "e" == unknown, a
           [2, 2, 0]],  # c c a
          features["source"])
      self.assertAllEqual([3, 3], features["source_sequence_length"])
      self.assertAllEqual(
          [[4, 2, 2],    # sos c c
           [4, 0, 1]],   # sos a b
          features["target_input"])
      self.assertAllEqual(
          [[2, 2, 3],   # c c eos
           [0, 1, 3]],  # a b eos
          features["target_output"])
      self.assertAllEqual([3, 3], features["target_sequence_length"])

      # Re-init iterator with skip_count=0.
      sess.run(iterator.initializer, feed_dict={skip_count: 0})

      features = sess.run(get_next)

      self.assertAllEqual(
          [[-1, -1, 0],  # "f" == unknown, "e" == unknown, a
           [2, 2, 0]],   # c c a
          features["source"])
      self.assertAllEqual([3, 3], features["source_sequence_length"])
      self.assertAllEqual(
          [[4, 2, 2],   # sos c c
           [4, 0, 1]],  # sos a b
          features["target_input"])
      self.assertAllEqual(
          [[2, 2, 3],   # c c eos
           [0, 1, 3]],  # a b eos
          features["target_output"])
      self.assertAllEqual([3, 3], features["target_sequence_length"])

  def testGetInferIterator(self):
    src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["c c a", "c a", "d", "f e a g"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        eos="eos",
        sos="sos")
    batch_size = 2
    dataset = iterator_utils.get_infer_iterator(
        src_dataset=src_dataset,
        src_vocab_table=src_vocab_table,
        batch_size=batch_size,
        eos=hparams.eos)
    table_initializer = tf.tables_initializer()
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer)
      features = sess.run(get_next)

      self.assertAllEqual(
          [
              [2, 2, 0],  # c c a
              [2, 0, 3]
          ],  # c a eos
          features["source"])
      self.assertAllEqual([3, 2], features["source_sequence_length"])


if __name__ == "__main__":
  tf.test.main()
