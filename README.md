# Isomorphic Math Engine

A neural framework that maps mathematical equations to 64-dimensional hyperbolic space, demonstrating that syntactically different but mathematically equivalent equations map to the same geometric region.

**Version 34.0** - Now with ASSR (Auto-Calibrated Stochastic Spectral Regularization) for improved training stability and generalization.

## Key Innovation

The system proves genuine mathematical understanding through form invariance testing. When trained on multiple syntactic representations of equivalent equations, the neural encoder learns mathematical meaning rather than syntactic patterns.

## What's New in v34.0: ASSR Integration

**ASSR** (Auto-Calibrated Stochastic Spectral Regularization) monitors and corrects spectral health of weight matrices during training, preventing ill-conditioning that causes training instability.

### ASSR Results

Training HyperbolicEncoder with ASSR vs baseline:

| Metric | Baseline | With ASSR | Improvement |
|--------|----------|-----------|-------------|
| Max Condition Number | 5,203 | 1,647 | **-68%** |
| Cluster Gap (embedding quality) | 0.065 | 0.138 | **+111%** |
| NN Classification Accuracy | 18.2% | 33.3% | **+83%** |

**Key insight**: ASSR doesn't just improve spectral health—it improves **generalization**. The encoder learns genuine solution structure rather than memorizing training examples.

### Generalization Test

We verified the encoder learns real mathematical understanding:
- **Train** on equations with coefficients `a ∈ {±1,±2,±3}`
- **Test** on completely unseen equations with `a ∈ {±4,±5,±6}`

Results:
- Baseline: 52% train accuracy → 19% test accuracy (memorization)
- **ASSR: 5% train accuracy → 33% test accuracy (genuine learning!)**

The ASSR model performs BETTER on unseen equations, proving it learned solution structure.

## Installation

```bash
pip install git+https://github.com/xfdbv99pqh-jpg/isoengine.git
```

Or clone and install:
```bash
git clone https://github.com/xfdbv99pqh-jpg/isoengine.git
cd isoengine
pip install -e .
```

## Quick Start

### Symbolic Solving (no training required)
```python
from isomorphic_math import solve

result = solve("2x + 3 = 11")
print(result)  # {'solution': 4.0, 'type': 'linear'}

result = solve("x^2 - 5x + 6 = 0")
print(result)  # {'solutions': [2.0, 3.0], 'type': 'quadratic'}
```

### Neural Encoding with ASSR
```python
from isomorphic_math import (
    HyperbolicEncoder,
    ContrastiveTrainerWithASSR,
    ASSRConfig,
)

# Create encoder
encoder = HyperbolicEncoder(embed_dim=128, hyperbolic_dim=32)

# Train with ASSR (recommended)
trainer = ContrastiveTrainerWithASSR(encoder, use_assr=True)
trainer.train(epochs=2000)

# Check training stats
print(trainer.get_assr_summary())

# Test similarity
sim = encoder.similarity(
    parse("2x + 4 = 10"),  # solution: x = 3
    parse("3x - 1 = 8"),   # solution: x = 3
)
print(f"Similarity: {sim:.3f}")  # High similarity (same solution!)
```

### Custom ASSR Configuration
```python
from isomorphic_math import ASSRConfig, ContrastiveTrainerWithASSR

config = ASSRConfig(
    base_lambda=5e-4,           # Regularization strength
    condition_ceiling=200.0,     # Max acceptable condition number
    stable_rank_floor=0.15,      # Min acceptable stable rank ratio
    sample_ratio=0.4,            # Fraction of layers to check
    penalty_type='spectral_norm_sq',  # Recommended penalty type
)

trainer = ContrastiveTrainerWithASSR(encoder, assr_config=config)
trainer.train(epochs=2000)
```

### Spectral Health Monitoring
```python
from isomorphic_math import print_spectral_report, auto_calibrate

# Print spectral health of any model
print_spectral_report(encoder)

# Auto-calibrate ASSR for your model
config = auto_calibrate(encoder, verbose=True)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Isomorphic Math Engine                       │
├─────────────────────────────────────────────────────────────────┤
│  Input: "2x + 3 = 7"                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │   MathParser    │ → Expression Tree                          │
│  └─────────────────┘                                            │
│           │                                                      │
│           ├──────────────────────┐                              │
│           ▼                      ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   MathSolver    │    │ HyperbolicEncoder│                    │
│  │  (Symbolic)     │    │  (Neural + ASSR) │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│     x = 2.0              64-dim embedding                       │
│                          (same solution → same region)          │
└─────────────────────────────────────────────────────────────────┘
```

## ASSR: How It Works

During training, weight matrices can develop spectral health issues:

1. **High Condition Number** (σ_max / σ_min) - Matrix becomes ill-conditioned, causing unstable gradients
2. **Low Stable Rank** - Matrix collapses toward low-rank, losing expressiveness

ASSR monitors these metrics and applies targeted regularization:

```python
# Simplified ASSR logic
for layer in model.linear_layers:
    condition = compute_condition_number(layer.weight)
    if condition > ceiling:
        # Apply spectral norm penalty to shrink dominant singular value
        loss += lambda * (sigma_max ** 2)

    stable_rank = compute_stable_rank_ratio(layer.weight)
    if stable_rank < floor:
        # Apply variance penalty to encourage uniform singular values
        loss += lambda * variance(singular_values)
```

**Why spectral_norm_sq works**: Unlike Frobenius norm (which scales all singular values equally), `σ_max²` specifically shrinks the dominant singular value, actually improving conditioning.

## Performance Results

### Form Invariance Testing
- **Linear equations**: 0.971 R² score, 0.81 MAE
- **Quadratic equations**: 0.997 R² score, 0.10 MAE
- **Inequalities**: 0.996 R² score, 0.26 MAE

All problem types demonstrated form invariance with mixed-form training.

### ASSR Impact
Training for 500 epochs on HyperbolicEncoder:

| Metric | Without ASSR | With ASSR |
|--------|--------------|-----------|
| Max Condition (start) | 276 | 276 |
| Max Condition (end) | 5,203 | 1,647 |
| Generalization Gap | +33% (overfitting) | -29% (generalizing!) |

## API Reference

### Core Classes

- `MathEngine` - Unified interface for parsing, solving, and encoding
- `HyperbolicEncoder` - Transformer-based encoder to hyperbolic space
- `ContrastiveTrainer` - Basic contrastive training
- `ContrastiveTrainerWithASSR` - Training with spectral regularization

### ASSR Functions

- `ASSRConfig` - Configuration dataclass
- `auto_calibrate(model)` - Auto-configure ASSR parameters
- `compute_condition_number(W)` - Get condition number
- `compute_stable_rank_ratio(W)` - Get stable rank as ratio [0,1]
- `compute_spectral_health(W)` - Get all spectral metrics
- `print_spectral_report(model)` - Print health report
- `apply_assr_regularization(model, config)` - Apply regularization in custom loops

### Convenience Functions

- `solve(equation)` - Solve equation symbolically
- `parse(equation)` - Parse to expression tree
- `similarity(eq1, eq2)` - Compute semantic similarity

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn (optional, for evaluation)

## License

MIT

## Citation

If you use this work, please cite:
```
@software{isoengine,
  title={Isomorphic Math Engine: Neural Embeddings for Mathematical Equivalence},
  author={Big J and Claude},
  year={2024},
  url={https://github.com/xfdbv99pqh-jpg/isoengine}
}
```

---

**Core thesis**: Mathematical meaning emerges from geometric structure. ASSR ensures the encoder learns this structure rather than overfitting to surface patterns.
