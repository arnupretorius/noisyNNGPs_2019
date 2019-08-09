# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Network Gaussian Process (nngp) kernel computation.

Implementaion based on
"Deep Neural Networks as Gaussian Processes" by
Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz,
Jeffrey Pennington, Jascha Sohl-Dickstein
arXiv:1711.00165 (https://arxiv.org/abs/1711.00165).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import tensorflow as tf

import interp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_precomputed_grid", True,
                     "Option to save/load pre-computed grid")
flags.DEFINE_integer(
    "fraction_of_int32", 32,
    "allow batches at most of size int32.max / fraction_of_int32")


class NNGPKernel(object):
  """The iterative covariance Kernel for Neural Network Gaussian Process.

  Args:
    depth: int, number of hidden layers in corresponding NN.
    nonlin_fn: tf ops corresponding to point-wise non-linearity in corresponding
      NN. e.g.) tf.nn.relu, tf.nn.sigmoid, lambda x: x * tf.nn.sigmoid(x), ...
    weight_var: initial value for the weight_variances parameter.
    bias_var: initial value for the bias_variance parameter.
    n_gauss: Number of gaussian integration grid. Choose odd integer, so that
      there is a gridpoint at 0.
    n_var: Number of variance grid points.
    n_corr: Number of correlation grid points.
    use_fixed_point_norm: bool, normalize input to variance fixed point.
      Defaults to False, normalizing input to unit norm over input dimension.
  """

  def __init__(self,
               depth=1,
               nonlin_fn=tf.tanh,
               weight_var=1.,
               bias_var=1.,
               mu_2=1.,
               n_gauss=101,
               n_var=151,
               n_corr=131,
               max_var=100,
               max_gauss=100,
               use_fixed_point_norm=False,
               grid_path=None,
               sess=None):
    self.depth = depth
    self.weight_var = weight_var
    self.bias_var = bias_var
    self.mu_2 = mu_2
    self.use_fixed_point_norm = use_fixed_point_norm
    self.sess = sess
    if FLAGS.use_precomputed_grid and (grid_path is None):
      raise ValueError("grid_path must be specified to use precomputed grid.")
    self.grid_path = grid_path


    self.nonlin_fn = nonlin_fn

    if self.use_fixed_point_norm:
      self.var_fixed_point_np, self.var_fixed_point = self.get_var_fixed_point()

  def get_var_fixed_point(self):
    with tf.name_scope("get_var_fixed_point"):
      # If normalized input length starts at 1.
      current_qaa = 0.5 * self.weight_var * tf.constant(
          [1.], dtype=tf.float64) * self.mu_2 + self.bias_var

      diff = 1.
      prev_qaa_np = 1.
      it = 0
      while diff > 1e-6 and it < 300:
        samp_qaa = 0.5 * self.weight_var * current_qaa * self.mu_2 + self.bias_var
        current_qaa = samp_qaa

        with tf.Session() as sess:
          current_qaa_np = sess.run(current_qaa)
        diff = np.abs(current_qaa_np - prev_qaa_np)
        it += 1
        prev_qaa_np = current_qaa_np
      return current_qaa_np, current_qaa

  def k_diag(self, input_x, return_full=True):
    """Iteratively building the diagonal part (variance) of the NNGP kernel.

    Args:
      input_x: tensor of input of size [num_data, input_dim].
      return_full: boolean for output to be [num_data] sized or a scalar value
        for normalized inputs

    Sets self.layer_qaa_dict of {layer #: qaa at the layer}

    Returns:
      qaa: variance at the output.
    """
    with tf.name_scope("Kdiag"):
      # If normalized input length starts at 1.
      if self.use_fixed_point_norm:
        current_qaa = self.var_fixed_point
      else:
        current_qaa = tf.convert_to_tensor([1.], dtype=tf.float64)
      self.layer_qaa_dict = {0: current_qaa}
      for l in xrange(self.depth):
        with tf.name_scope("layer_%d" % l):
          samp_qaa = 0.5 * self.weight_var * current_qaa * self.mu_2 + self.bias_var
          self.layer_qaa_dict[l + 1] = samp_qaa
          current_qaa = samp_qaa

      if return_full:
        qaa = tf.tile(current_qaa[:1], ([input_x.shape[0].value]))
      else:
        qaa = current_qaa[0]
      return qaa

  def k_full(self, input1, input2=None):
    """Iteratively building the full NNGP kernel.
    """
    input1 = self._input_layer_normalization(input1)
    if input2 is None:
      input2 = input1
    else:
      input2 = self._input_layer_normalization(input2)

    with tf.name_scope("k_full"):
      cov_init = tf.matmul(
          input1, input2, transpose_b=True) / input1.shape[1].value
      
      self.k_diag(input1)
      q_aa_init = self.layer_qaa_dict[0]

      q_ab = cov_init
      corr = q_ab / q_aa_init[0]
      corr_init = corr
      self.layer_corr_dict = {0: corr}

      if FLAGS.fraction_of_int32 > 1:
        batch_size, batch_count = self._get_batch_size_and_count(input1, input2)
        with tf.name_scope("q_ab"):
          q_ab_all = []
          for b_x in range(batch_count):
            with tf.name_scope("batch_%d" % b_x):
              corr_flat_batch = corr[
                  batch_size * b_x : batch_size * (b_x + 1), :]
              corr_flat_batch = tf.reshape(corr_flat_batch, [-1])

              for l in xrange(self.depth):
                with tf.name_scope("layer_%d" % l):
                  q_aa = self.layer_qaa_dict[l]
                  multiplier = tf.constant(10**8, dtype=tf.float64)
                  corr = tf.round(corr * multiplier) / multiplier
                  q_ab = (corr*tf.math.asin(corr) + tf.math.sqrt(1-tf.math.pow(corr, 2)))/np.pi + corr/2
                  q_ab = 0.5 * self.weight_var * q_ab + self.bias_var
                  corr_flat_batch = q_ab / self.layer_qaa_dict[l + 1][0]
                  corr = corr_flat_batch
                  self.layer_corr_dict[l+1] = corr
                  

              q_ab_all.append(q_ab)

          q_ab_all = tf.parallel_stack(q_ab_all)
      else:
        with tf.name_scope("q_ab"):
          corr_flat = tf.reshape(corr, [-1])
          for l in xrange(self.depth):
            with tf.name_scope("layer_%d" % l):
              q_aa = self.layer_qaa_dict[l]
              multiplier = tf.constant(10**8, dtype=tf.float64)
              corr = tf.round(corr * multiplier) / multiplier
              q_ab = (corr*tf.math.asin(corr) + tf.math.sqrt(1-tf.math.pow(corr, 2)))/np.pi + corr/2
              q_ab = 0.5 * self.weight_var * q_ab + self.bias_var
              corr_flat = q_ab / self.layer_qaa_dict[l+1][0]
              corr = corr_flat
            q_ab_all = q_ab

    return  tf.reshape(q_ab_all, cov_init.shape, "qab")

  def _input_layer_normalization(self, x):
    """Input normalization to unit variance or fixed point variance.
    """
    with tf.name_scope("input_layer_normalization"):
      # Layer norm, fix to unit variance
      eps = 1e-15
      mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
      x_normalized = (x - mean) / tf.sqrt(var + eps)
      if self.use_fixed_point_norm:
        x_normalized *= tf.sqrt(
            (self.var_fixed_point[0] - self.bias_var) / self.weight_var)
      return x_normalized

  def _get_batch_size_and_count(self, input1, input2):
    """Compute batch size and number to split when input size is large.

    Args:
      input1: tensor, input tensor to covariance matrix
      input2: tensor, second input tensor to covariance matrix

    Returns:
      batch_size: int, size of each batch
      batch_count: int, number of batches
    """
    input1_size = input1.shape[0].value
    input2_size = input2.shape[0].value

    batch_size = min(np.iinfo(np.int32).max //
                     (FLAGS.fraction_of_int32 * input2_size), input1_size)
    while input1_size % batch_size != 0:
      batch_size -= 1

    batch_count = input1_size // batch_size
    return batch_size, batch_count