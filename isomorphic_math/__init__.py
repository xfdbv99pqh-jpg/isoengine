"""
Isomorphic Math Engine
======================
Geometric embeddings for mathematical equations.

Usage:
    from isomorphic_math import MathEngine, solve, parse

    # Quick solve
    result = solve("2x + 3 = 11")

    # Full engine with neural encoder
    engine = MathEngine()
    engine.train(epochs=1000)
    similarity = engine.similarity("2x = 6", "x + 1 = 4")

    # Multi-head solution prediction (best results)
    from isomorphic_math import HyperbolicEncoder, MultiHeadTrainer
    encoder = HyperbolicEncoder()
    trainer = MultiHeadTrainer(encoder, device)
    trainer.train(epochs=3000)
    prediction = trainer.predict_linear([equation])

Author: Big J + Claude
Version: 33.0
"""

from .core import (
    # Expression system
    Op, Expr, ProblemType,
    Const, VarX, VarY, VarZ,
    Add, Sub, Mul, Div, Pow, Neg,
    Eq, Gt, Lt, Gte, Lte,
    System, Or, And,
    Sin, Cos, Tan, Ln, Exp, Sqrt, Abs,
    Deriv,

    # Parser
    MathParser,

    # Solvers
    LinearSolver, QuadraticSolver, SystemSolver, InequalitySolver,
    MathSolver,

    # Calculus
    Differentiator, Simplifier,

    # Generators
    ProblemGenerator,
)

from .encoder import (
    HyperbolicEncoder,
    ContrastiveTrainer,
    SolutionPredictor,
)

from .multihead import MultiHeadTrainer

from .engine import MathEngine

# Convenience functions
_default_engine = None

def get_engine():
    """Get or create default engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MathEngine()
    return _default_engine

def solve(problem):
    """Quick solve any math problem."""
    return get_engine().solve(problem)

def parse(text):
    """Parse string to expression tree."""
    return get_engine().parse(text)

def differentiate(expr):
    """Compute derivative."""
    return get_engine().differentiate(expr)

def encode(exprs):
    """Encode expressions (requires trained encoder)."""
    return get_engine().encode(exprs)

def similarity(expr1, expr2):
    """Compare two expressions (requires trained encoder)."""
    return get_engine().similarity(expr1, expr2)

__version__ = "33.0"
__all__ = [
    # Core
    'Op', 'Expr', 'ProblemType',
    'Const', 'VarX', 'VarY', 'VarZ',
    'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Neg',
    'Eq', 'Gt', 'Lt', 'Gte', 'Lte',
    'System', 'Or', 'And',
    'Sin', 'Cos', 'Tan', 'Ln', 'Exp', 'Sqrt', 'Abs',
    'Deriv',

    # Classes
    'MathParser', 'MathSolver', 'MathEngine',
    'LinearSolver', 'QuadraticSolver', 'SystemSolver', 'InequalitySolver',
    'Differentiator', 'Simplifier',
    'ProblemGenerator',
    'HyperbolicEncoder', 'ContrastiveTrainer', 'SolutionPredictor',
    'MultiHeadTrainer',

    # Functions
    'solve', 'parse', 'differentiate', 'encode', 'similarity',
    'get_engine',
]
