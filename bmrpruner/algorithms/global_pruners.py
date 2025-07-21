# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
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

"""This file implements common global pruning algorithms."""
import dataclasses
from typing import Optional, Dict

import chex
import flax
import jax
import jax.numpy as jnp
from bmrpruner import base_updater
from bmrpruner import sparsity_distributions
from bmrpruner import sparsity_types
from bmrpruner.algorithms import pruners
from bmrpruner.algorithms import bmr_pruners


@dataclasses.dataclass
class GlobalPruningMixin:
  """Implements Mixin for global pruning."""

  custom_sparsity_map: Optional[Dict[str, float]] = None
  filter_fn: base_updater.FilterFnType = (
      sparsity_distributions.NOT_DIM_ONE_FILTER_FN
  )
  normalization_eps: float = 1e-8
  use_normalization: bool = True
  sparsity: float = 0.0

  def init_state(self, params):
    """Regular init. but target_sparsities are set to orignal sparsity."""
    if not isinstance(self.sparsity_type, sparsity_types.Unstructured):
      raise AttributeError(
          f'Sparsity type {self.sparsity_type.__class__} is not supported for'
          ' GlobalPruningMixin.'
      )
    masks = self.create_masks(params, 0.0)
    if self.use_packed_masks:
      masks = jax.tree.map(jnp.packbits, masks)
    # Global sparsity only needs global target sparsity.
    return base_updater.SparseState(
        masks=masks,
        target_sparsities=self.sparsity,
        count=jnp.zeros([], jnp.int32),
    )

  def instant_sparsify(
      self, params, grads = None, **kwargs
  ):
    # Global sparsity doesn't require sparsity distribution as it uses
    # global ordering of scores.
    scores = self.calculate_scores(params, grads=grads, **kwargs)
    masks = self.create_masks(scores, self.sparsity)
    if self.use_packed_masks:
      masks = jax.tree.map(jnp.packbits, masks)
    return self.apply_masks(params, masks, is_packed=False), masks

  def create_masks(self, scores, target_sparsity):
    """Creates masks using global ordering."""
    custom_sparsity_map = self.custom_sparsity_map or {}

    def _unified_filter_fn(k, score):
      return self.filter_fn(k, score) and k not in custom_sparsity_map

    def _maybe_normalize(score):
      if self.use_normalization:
        return score / (jnp.linalg.norm(score) + self.normalization_eps)
      else:
        return score

    def _func1(path, score):
      if _unified_filter_fn(path, score):
        return _maybe_normalize(score)

    filtered_scores = jax.tree.map_with_path(
      lambda path, score: _func1(path, score),
      scores
    )

    leaves, tree = jax.tree.flatten(filtered_scores)
    filtered_scores_concat = jnp.concatenate(leaves, axis=None)

    flat_mask_concat = self.topk_fn(filtered_scores_concat, target_sparsity)

    res = []
    cur_index = 0
    for param in leaves:
      next_index = cur_index + param.size
      flat_mask = flat_mask_concat[cur_index:next_index]
      res.append(jnp.reshape(flat_mask, param.shape))
      cur_index = next_index

    mask1 = jax.tree.unflatten(tree, res)
    
    def _func2(path, score):
      if path in custom_sparsity_map:
        return self.topk_fn(score, custom_sparsity_map[path])
      
    mask2 = jax.tree.map_with_path(lambda path, score: _func2(path, score), scores)
    
    def _func3(a, b):
      if a is not None:
        return a
      elif b is not None:
        return b
      
    return jax.tree.map(
      lambda a, b: _func3(a, b), mask1, mask2, is_leaf=lambda x: x is None
    )


@dataclasses.dataclass
class GlobalMagnitudePruning(GlobalPruningMixin, pruners.MagnitudePruning):
  """Magnitude pruner with global ordering."""

  pass


@dataclasses.dataclass
class GlobalSaliencyPruning(GlobalPruningMixin, pruners.SaliencyPruning):
  """Saliency pruner with global ordering."""

  pass

@dataclasses.dataclass
class GlobalBMRPruning(GlobalPruningMixin, bmr_pruners.BMRPruning):
  """BMR pruner with global ordering."""

  pass
