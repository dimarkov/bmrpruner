# coding=utf-8
# Copyright 2024 BMRPruner Authors.
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

"""This file implements Bayesian Model Reduction pruning algorithms."""
import dataclasses
import jax
import jax.numpy as jnp
from bmrpruner import base_updater

BaseUpdater = base_updater.BaseUpdater


@dataclasses.dataclass
class BMRPruning(BaseUpdater):
  """Implements Bayesian Model Reduction based pruning.

  This pruner calculates scores based on the mean and variance of the
  parameter's posterior distribution, as estimated by an optimizer like IVON.
  The score is calculated as `abs(mean) / variance`.
  """

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del grads
    
    # The IVON state is the first element in the inner_state tuple
    state = sparse_state.inner_state[0]
    
    if not hasattr(state, 'hess'):
        raise AttributeError(
            'The wrapped optimizer state must have a `hess` attribute '
            'to use BMRPruning. Please use an IVON-compatible optimizer.'
        )
    
    # Get variance from hessian.
    variance = jax.tree.map(lambda h: state.ess / (h + state.weight_decay), state.hess)
    
    # Score is magnitude / variance. High score = important.
    scores = jax.tree.map(
        lambda mean, var: jnp.abs(mean) / (var + 1e-8), params, variance
    )
    return scores


@dataclasses.dataclass
class AdaptiveBMRPruning(BMRPruning):
  """Implements Adaptive Bayesian Model Reduction based pruning."""

  pass
