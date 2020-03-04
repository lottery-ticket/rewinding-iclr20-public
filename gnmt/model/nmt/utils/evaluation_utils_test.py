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

"""Tests for evaluation_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import evaluation_utils


class EvaluationUtilsTest(tf.test.TestCase):

  def testEvaluate(self):
    output = "third_party/tensorflow_models/mlperf/models/rough/nmt/testdata/deen_output"
    ref_bpe = "third_party/tensorflow_models/mlperf/models/rough/nmt/testdata/deen_ref_bpe"
    ref_spm = "third_party/tensorflow_models/mlperf/models/rough/nmt/testdata/deen_ref_spm"

    expected_bleu_score = 22.5855084573

    bpe_bleu_score = evaluation_utils.evaluate(
        ref_bpe, output, "bleu", "bpe")

    self.assertAlmostEqual(expected_bleu_score, bpe_bleu_score)

    spm_bleu_score = evaluation_utils.evaluate(
        ref_spm, output, "bleu", "spm")

    self.assertAlmostEqual(expected_bleu_score, spm_bleu_score)

if __name__ == "__main__":
  tf.test.main()
