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

"""Algorithms implemented in bmrpruner."""


from bmrpruner.algorithms.global_pruners import GlobalMagnitudePruning
from bmrpruner.algorithms.global_pruners import GlobalSaliencyPruning
from bmrpruner.algorithms.global_pruners import GlobalBMRPruning
from bmrpruner.algorithms.pruners import MagnitudePruning
from bmrpruner.algorithms.pruners import RandomPruning
from bmrpruner.algorithms.pruners import SaliencyPruning
from bmrpruner.algorithms.sparse_trainers import RigL
from bmrpruner.algorithms.sparse_trainers import SET
from bmrpruner.algorithms.sparse_trainers import StaticRandomSparse
from bmrpruner.algorithms.ste import SteMagnitudePruning
from bmrpruner.algorithms.ste import SteRandomPruning
from bmrpruner.algorithms.bmr_pruners import BMRPruning
