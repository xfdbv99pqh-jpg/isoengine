"""
Main MathEngine class - unified interface for all operations.
"""

from typing import Union, List, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .core import (
    Expr, Op,
    MathParser, MathSolver, Differentiator,
)
from .encoder import HyperbolicEncoder, ContrastiveTrainer


class MathEngine:
    """
    Unified interface for the Isomorphic Math Engine.
    
    Usage:
        engine = MathEngine()
        
        # Solve problems
        result = engine.solve("2x + 3 = 11")
        result = engine.solve("x^2 - 5x + 6 = 0")
        result = engine.solve("2x + 3 > 7")
        result = engine.solve("d/dx(x^3 + sin(x))")
        
        # Check similarity (requires trained encoder)
        engine.train(epochs=1000)
        sim = engine.similarity("2x + 4 = 10", "x + 2 = 5")  # Both have x=3
        
        # Get embeddings
        embeddings = engine.encode(["x + 1 = 5", "x^2 = 4"])
    """
    
    def __init__(self):
        self.parser = MathParser()
        self.solver = MathSolver()
        self.differentiator = Differentiator()
        self.encoder = None
        self.trainer = None
        self._device = None
    
    @property
    def device(self):
        if self._device is None and TORCH_AVAILABLE:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def solve(self, problem: Union[str, Expr]) -> Dict:
        """Solve any supported math problem."""
        return self.solver.solve(problem)
    
    def parse(self, text: str) -> Expr:
        """Parse string to expression tree."""
        return self.parser.parse(text)
    
    def differentiate(self, expr: Union[str, Expr]) -> str:
        """Compute derivative."""
        if isinstance(expr, str):
            expr = self.parser.parse(expr)
        result = self.differentiator.differentiate(expr)
        return str(result)
    
    def train(self, epochs: int = 2000, verbose: bool = True):
        """Train the neural encoder."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for neural encoder")
        
        self.encoder = HyperbolicEncoder().to(self.device)
        self.trainer = ContrastiveTrainer(self.encoder, self.device)
        self.trainer.train(epochs=epochs, verbose=verbose)
        
        # Freeze encoder after training
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def similarity(self, expr1: Union[str, Expr], expr2: Union[str, Expr]) -> float:
        """
        Compute similarity between two expressions.
        Expressions with same solution have high similarity.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not trained. Call train() first.")
        
        if isinstance(expr1, str):
            expr1 = self.parser.parse(expr1)
        if isinstance(expr2, str):
            expr2 = self.parser.parse(expr2)
        
        return self.encoder.similarity(expr1, expr2)
    
    def encode(self, exprs: Union[str, Expr, List]) -> 'torch.Tensor':
        """Encode expressions to embedding vectors."""
        if self.encoder is None:
            raise RuntimeError("Encoder not trained. Call train() first.")
        
        if isinstance(exprs, str):
            exprs = [self.parser.parse(exprs)]
        elif isinstance(exprs, Expr):
            exprs = [exprs]
        else:
            exprs = [self.parser.parse(e) if isinstance(e, str) else e for e in exprs]
        
        with torch.no_grad():
            return self.encoder.encode_exprs(exprs, self.device)
    
    def is_valid(self, expr: Union[str, Expr]) -> bool:
        """Check if expression is mathematically valid."""
        try:
            if isinstance(expr, str):
                expr = self.parser.parse(expr)
            return not self._has_invalid(expr)
        except Exception:
            return False
    
    def _has_invalid(self, expr: Expr) -> bool:
        if expr is None:
            return False
        if expr.op == Op.DIV:
            if expr.right and expr.right.op == Op.CONST and expr.right.value == 0:
                return True
        return self._has_invalid(expr.left) or self._has_invalid(expr.right)
    
    def save(self, path: str):
        """Save trained encoder to file."""
        if self.encoder is None:
            raise RuntimeError("No encoder to save. Call train() first.")
        torch.save(self.encoder.state_dict(), path)
        print(f"Saved encoder to {path}")
    
    def load(self, path: str):
        """Load trained encoder from file."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        self.encoder = HyperbolicEncoder().to(self.device)
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.encoder.eval()
        print(f"Loaded encoder from {path}")
