# coding=utf-8
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

#!/bin/bash


python -m bmrpruner.base_updater_test
python -m bmrpruner.mask_calculator_test
python -m bmrpruner.sparsity_distributions_test
python -m bmrpruner.sparsity_schedules_test
python -m bmrpruner.base_updater_test
python -m bmrpruner.utils_test
python -m bmrpruner.algorithms.global_pruners_test
python -m bmrpruner.algorithms.pruners_test
python -m bmrpruner.algorithms.sparse_trainers_test
python -m bmrpruner.algorithms.ste_test
