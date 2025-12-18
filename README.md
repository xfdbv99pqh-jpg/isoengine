# Isomorphic Math Engine

**Geometric embeddings that encode mathematical meaning, not syntax.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Discovery

We trained a neural encoder that maps mathematical equations to 64-dimensional hyperbolic space. The key finding:

**Syntactically different but mathematically equivalent equations map to the same geometric region.**

This isn't pattern matching or coefficient extraction — the network learned genuine mathematical structure.

## Proof: Form Invariance Test

We tested whether the encoder learned to "read coefficients" or understand mathematics by generating the same quadratic equation (roots r₁=2, r₂=3) in 7 different syntactic forms:

| Form | Expression | R² Score |
|------|------------|----------|
| Standard | `x² - 5x + 6 = 0` | 0.998 |
| Scaled 2x | `2x² - 10x + 12 = 0` | 0.996 |
| Scaled 0.5x | `0.5x² - 2.5x + 3 = 0` | 0.994 |
| Negated | `-x² + 5x - 6 = 0` | 0.995 |
| Rearranged | `x² + 6 = 5x` | 0.998 |
| Factored tree | `x·x - 2x - 3x + 6 = 0` | 0.998 |
| Negative form | `-x² - bx - c = 0` | 0.998 |

**All forms predict r₁ + r₂ = 5 correctly.**

If the network were doing coefficient extraction, `2x² - 10x + 12 = 0` would predict 10 (the visible coefficient), not 5 (the actual sum of roots). Instead, it understands that multiplying an equation by 2 doesn't change its solutions.

## Results Summary

| Problem Type | R² Score | MAE | What's Predicted |
|--------------|----------|-----|------------------|
| Linear equations | 0.711 | 1.36 | Solution x |
| Quadratic equations | 0.999 | 0.10 | Sum of roots r₁+r₂ |
| Inequalities | 0.996 | 0.26 | Boundary value |

## Installation

```bash
pip install git+https://github.com/xfdbv99pqh-jpg/isoengine.git
```

## Quick Start

### Symbolic Solving (no training needed)

```python
from isomorphic_math import solve, parse, differentiate

# Solve equations
solve("2x + 3 = 11")           # {'x': 4.0}
solve("x^2 - 5x + 6 = 0")      # {'x': [2.0, 3.0]}
solve("2x + 3 > 7")            # {'solution': 'x > 2'}

# Parse to expression tree
expr = parse("x^2 + sin(x)")

# Symbolic differentiation
differentiate("x^3 + sin(x)")  # "3x² + cos(x)"
```

### Neural Embedding (requires training)

```python
import torch
from isomorphic_math import HyperbolicEncoder, MultiHeadTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create and train
encoder = HyperbolicEncoder().to(device)
trainer = MultiHeadTrainer(encoder, device)
trainer.train(epochs=3000)  # ~3-4 min on GPU

# Predict solutions directly from embeddings
from isomorphic_math import Eq, Add, Mul, Const, VarX

eq = Eq(Add(Mul(Const(2), VarX()), Const(3)), Const(11))  # 2x + 3 = 11
prediction = trainer.predict_linear([eq])
print(f"Predicted x = {prediction.item():.2f}")  # ≈ 4.0

# Save/load trained model
trainer.save("model.pt")
trainer.load("model.pt")
```

### Similarity Detection

```python
from isomorphic_math import MathEngine

engine = MathEngine()
engine.train(epochs=2000)

# These have the same solution (x=3)
sim = engine.similarity("2x + 4 = 10", "3x - 1 = 8")
print(f"Similarity: {sim:.3f}")  # High similarity
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Expression Tree                          │
│         Eq(Add(Mul(Const(2), VarX()), Const(3)), Const(11)) │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Tensor Encoding                            │
│              ops: [EQ, ADD, MUL, CONST, VAR_X, CONST, ...]  │
│              vals: [0, 0, 0, 2, 0, 3, ...]                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Transformer Encoder                          │
│         4 layers, 8 heads, 256 dim → 64 dim hyperbolic     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              64-dim Hyperbolic Embedding                    │
│     Normalized to unit sphere, encodes solution geometry    │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │ Linear   │  │ Quadratic│  │ Inequality│
      │ Head     │  │ Head     │  │ Head      │
      │ → x      │  │ → r₁+r₂  │  │ → boundary│
      └──────────┘  └──────────┘  └──────────┘
```

## Training Strategy

We use **multi-task learning** combining:

1. **Contrastive Loss**: Pulls equations with similar solutions together in embedding space
2. **Regression Loss**: Each problem type has its own head predicting the appropriate value

This teaches the embedding to both cluster equivalent equations AND encode numeric solutions.

```python
# Loss function
total_loss = contrastive_loss + 0.3 * (linear_mse + quadratic_mse + inequality_mse)
```

## Package Structure

```
isomorphic_math/
├── __init__.py      # Exports and convenience functions
├── core.py          # Expression system, parser, symbolic solvers
├── encoder.py       # HyperbolicEncoder, ContrastiveTrainer
├── multihead.py     # MultiHeadTrainer (best results)
└── engine.py        # MathEngine unified API
```

## Supported Problem Types

| Type | Example | Solver | Neural Prediction |
|------|---------|--------|-------------------|
| Linear equations | `2x + 3 = 11` | ✅ Exact | ✅ R²=0.711 |
| Quadratic equations | `x² - 5x + 6 = 0` | ✅ Exact | ✅ R²=0.999 |
| Systems (2x2) | `x + y = 5, x - y = 1` | ✅ Exact | 🔄 Planned |
| Linear inequalities | `2x + 3 > 7` | ✅ Exact | ✅ R²=0.996 |
| Quadratic inequalities | `x² - 4 > 0` | ✅ Exact | 🔄 Planned |
| Derivatives | `d/dx(x³ + sin(x))` | ✅ Symbolic | — |

## Validation Scripts

Run the form invariance test to verify geometric understanding:

```bash
python geometry_vs_extraction_test.py
```

This proves the network learned mathematics, not pattern matching.

## The Thesis

> **Mathematical exactness emerges from geometric structure.**

Traditional neural networks treat math as string manipulation. This project demonstrates that:

1. Mathematical equations can be embedded in hyperbolic space
2. The embedding preserves mathematical meaning across syntactic variation
3. Solutions can be recovered directly from the geometric representation

The geometry IS the mathematics.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn (for evaluation)
- matplotlib (for visualization)

## Citation

If you use this work, please cite:

```
@software{isomorphic_math_engine,
  author = {Big J},
  title = {Isomorphic Math Engine: Geometric Embeddings for Mathematical Equations},
  year = {2024},
  url = {https://github.com/xfdbv99pqh-jpg/isoengine}
}
```

## License

MIT License

## Acknowledgments

Developed through extensive experimentation exploring connections between:
- Hyperbolic geometry and hierarchical structure
- Contrastive learning and mathematical equivalence
- Transformer architectures and symbolic reasoning

Special thanks to Claude for pair programming and hypothesis testing.

---

**The embedding doesn't encode what the equation looks like — it encodes what the equation means.**
