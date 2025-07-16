# BMRPruner: Bayesian Model Reduction for Neural Networks

*A fork of [JaxPruner](https://github.com/google-research/jaxpruner) extended for Bayesian Model Reduction*

## Overview

BMRPruner extends the excellent JaxPruner library to support **Bayesian Model Reduction (BMR)**, a principled approach to pruning approximate posteriors of deep neural networks. Unlike traditional pruning methods that work with point estimates, BMR leverages uncertainty information to make more informed pruning decisions.

### Key Features

- 🎯 **Bayesian Model Reduction**: Uncertainty-aware pruning of posterior distributions
- 🔄 **JaxPruner Compatibility**: All original algorithms and APIs remain functional
- 🌳 **Generic PyTree Support**: Works with Flax, Equinox, and any JAX-based models
- 📊 **BLRAX Integration**: Seamless integration with posterior-tracking optimizers
- 🔧 **Easy Migration**: Drop-in replacement for existing JaxPruner workflows
- ⚡ **Minimal Overhead**: Maintains the performance characteristics of the original library

## What is Bayesian Model Reduction (BMR)?

Traditional pruning methods remove parameters based on magnitude or gradient information, treating neural network weights as point estimates. **Bayesian Model Reduction** takes a fundamentally different approach by working with approximate posterior distributions over parameters.

### Key Differences:

| Traditional Pruning | Bayesian Model Reduction |
|-------------------|-------------------------|
| Works with point estimates (single weight values) | Works with posterior distributions (mean + uncertainty) |
| Uses magnitude or gradient-based importance | Uses uncertainty-aware importance scoring |
| Binary decisions (keep/remove) | Gradual uncertainty reduction |
| No calibrated uncertainty | Maintains calibrated predictions |

### Why BMR?

- **Principled Uncertainty**: Leverages the natural uncertainty in neural network parameters
- **Better Calibration**: Maintains well-calibrated predictions after pruning
- **Informed Decisions**: Uses both magnitude and uncertainty for importance scoring
- **Gradual Reduction**: Smoothly reduces model complexity while preserving performance

## Requirements

### For Bayesian Model Reduction
BMR requires optimizers that track posterior estimates (both mean and uncertainty):
- **[BLRAX Optimizer](https://github.com/dimarkov/blrax)** (recommended)
- Other variational inference optimizers that maintain posterior distributions

### For Traditional Pruning (Backward Compatible)
- Any Optax optimizer
- All original JaxPruner functionality remains available

## Installation

```bash
# Install BMRPruner
pip install git+https://github.com/your-username/bmrpruner.git

# For BMR functionality, also install BLRAX
pip install git+https://github.com/dimarkov/blrax.git
```

Alternatively, clone and install from source:

```bash
git clone https://github.com/your-username/bmrpruner.git
cd bmrpruner
bash run.sh
```

## Quick Start

### Traditional Pruning (JaxPruner Compatible)

BMRPruner maintains full backward compatibility with JaxPruner:

```python
import bmrpruner

# Existing JaxPruner code works unchanged
tx, params = _existing_code()
pruner = bmrpruner.MagnitudePruning(**config)
tx = pruner.wrap_optax(tx)
```

### Bayesian Model Reduction with BLRAX

```python
import blrax
import bmrpruner
import ml_collections

# Setup BLRAX optimizer for posterior tracking
optimizer = blrax.ivon(lr=0.01, ess=1000, hess_init=0.1)

config = ml_collections.ConfigDict()
config.algorithm = 'bmr'
config.update_freq = 10
config.update_end_step = 2000
config.update_start_step = 500
config.sparsity = 0.95
config.dist_type = 'erk'

# create BMR pruner
sparsity_updater = bmrpruner.create_updater_from_config(config)
optimizer = sparsity_updater.wrap_optax(optimizer)
optstate = optimizer.init(params)

# Training loop with posterior sampling
for step in range(num_steps):
    # Sample from posterior
    loss, grads, inner_state = blrax.noisy_value_and_grad(
      loss_fn,
      optstate.inner_state,
      params,
      rng_key
      mc_samples=mc_samples,
      mask=optstate.masks
    )

    optstate = optstate._replace(inner_state)
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(param_sample, batch)
    
    # Update with BMR
    updates, optstate = tx.update(grads, optstate, params)
    params = optax.apply_updates(params, updates)
```

### TODO: Generic PyTree Support (Equinox Example)

```python
import equinox as eqx
import bmrpruner
import jax.random as jr

# Works with any PyTree structure
depth = 3
in_size = 28 * 28
out_size = 10
num_neurons = 50
model = eqx.nn.MLP(in_size, out_size, num_neurons, depth, key=jr.PRNGKey(0))

# BMRPruner works seamlessly with Equinox models
pruner = bmrpruner.MagnitudePruning(sparsity=0.9)
tx = pruner.wrap_optax(optax.adam(1e-3))
```

## BMR Algorithms

BMRPruner introduces new algorithms specifically designed for Bayesian Model Reduction:

- **`UncertaintyMagnitudePruning`**: Combines magnitude and uncertainty for importance scoring
- **`PosteriorVariancePruning`**: Prunes parameters with low posterior variance
- **`BayesianGlobalPruning`**: Global pruning using posterior information
- **`AdaptiveBMR`**: Dynamically adjusts pruning based on uncertainty evolution

## Integration Examples

BMRPruner maintains JaxPruner's philosophy of easy integration with popular JAX libraries:

- **[Flax](https://github.com/google/flax)**: Neural networks and transformers
- **[Equinox](https://github.com/patrick-kidger/equinox)**: Differentiable programming
- **[Scenic](https://github.com/google-research/scenic)**: Computer vision models
- **[T5X](https://github.com/google-research/t5x)**: Large language models
- **[FedJAX](https://github.com/google/fedjax)**: Federated learning

## Baselines and Results

### Traditional Pruning (JaxPruner Compatible)
All original JaxPruner baselines are maintained:

|        |   no_prune |     random |   magnitude |   saliency |   global_magnitude |   magnitude_ste |   static_sparse |        set |       rigl |
|:-------|-----------:|-----------:|------------:|-----------:|-------------------:|----------------:|----------------:|-----------:|-----------:|
| ResNet-50 |   76.67    |   70.192   |    75.532   |   74.93    |           75.486   |         73.542  |        71.344   |   74.566   |   74.752   |
| ViT-B/16 (90ep)  |   74.044   |   69.756   |    72.892   |   72.802   |           73.598   |         74.208  |        64.61    |   70.982   |   71.582   |

### Bayesian Model Reduction Results
*Coming soon: Comprehensive benchmarks comparing BMR with traditional pruning methods*

## Migration from JaxPruner

Migrating from JaxPruner to BMRPruner is straightforward:

1. **Replace imports**: `import jaxpruner` → `import bmrpruner`
2. **Existing code works unchanged**: All JaxPruner APIs are preserved
3. **Optional BMR upgrade**: Add IVON optimizer and BMR algorithms when ready

## Contributing

We welcome contributions! BMRPruner builds on the solid foundation of JaxPruner while extending it for Bayesian methods. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

<!-- ## Citation

If you use BMRPruner in your research, please cite both the original JaxPruner paper and the BLRAX optimizer:

```bibtex
@inproceedings{jaxpruner,
  title={JaxPruner: A concise library for sparsity research},
  author={Joo Hyung Lee and Wonpyo Park and Nicole Mitchell and Jonathan Pilault and Johan S. Obando-Ceron and Han-Byul Kim and Namhoon Lee and Elias Frantar and Yun Long and Amir Yazdanbakhsh and Shivani Agrawal and Suvinay Subramanian and Xin Wang and Sheng-Chun Kao and Xingyao Zhang and Trevor Gale and Aart J. C. Bik and Woohyun Han and Milen Ferev and Zhonglin Han and Hong-Seok Kim and Yann Dauphin and Karolina Dziugaite and Pablo Samuel Castro and Utku Evci},
  year={2023}
}

@inproceedings{blrax,
  title={BLRAX: A Bayesian Library for Ranking, Aggregation, and Explanation},
  author={D. Markov and M. E. Khan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
``` -->

## Acknowledgments

BMRPruner is built upon the excellent [JaxPruner](https://github.com/google-research/jaxpruner) library. We thank the original authors for their foundational work in JAX-based sparsity research. The Bayesian Model Reduction capabilities are enabled by the [IVON optimizer](https://arxiv.org/abs/2402.17641) for variational inference as implemented in [blrax](https://github.com/dimarkov/blrax) repo.

## License and Attribution

This project is a derivative work of [JaxPruner](https://github.com/google-research/jaxpruner) and is licensed under the Apache License, Version 2.0, the same license as the original work.

**Original Work Attribution:**
- Original JaxPruner: Copyright 2024 Jaxpruner Authors (Google Research)
- This derivative work (BMRPruner): Copyright 2024 BMRPruner Contributors

**Major Modifications Made:**
- Package renamed from 'jaxpruner' to 'bmrpruner'
- Added support for Bayesian Model Reduction algorithms
- Extended documentation for BLRAX optimizer integration
- Added support for generic PyTree structures (Equinox, etc.)
- Maintained backward compatibility with original JaxPruner APIs

See the [NOTICE](NOTICE) file for complete attribution details.

## Disclaimer

This is a research library extending JaxPruner for Bayesian Model Reduction. While we maintain backward compatibility, this is not an officially supported Google product and is not affiliated with Google in any way.
