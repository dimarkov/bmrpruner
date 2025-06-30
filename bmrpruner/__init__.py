# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
# Copyright 2024 BMRPruner Contributors.
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
#
# This file is part of BMRPruner, a derivative work of JaxPruner.
# Original JaxPruner: https://github.com/google-research/jaxpruner

"""APIs for bmrpruner."""

from bmrpruner.algorithms import *


from bmrpruner.api import ALGORITHM_REGISTRY
from bmrpruner.api import all_algorithm_names
from bmrpruner.api import create_updater_from_config
from bmrpruner.api import register_algorithm
from bmrpruner.base_updater import apply_mask
from bmrpruner.base_updater import BaseUpdater
from bmrpruner.base_updater import NoPruning
from bmrpruner.base_updater import SparseState
from bmrpruner.sparsity_schedules import NoUpdateSchedule
from bmrpruner.sparsity_schedules import OneShotSchedule
from bmrpruner.sparsity_schedules import PeriodicSchedule
from bmrpruner.sparsity_schedules import PolynomialSchedule
from bmrpruner.sparsity_types import SparsityType
from bmrpruner.utils import summarize_intersection
from bmrpruner.utils import summarize_sparsity
