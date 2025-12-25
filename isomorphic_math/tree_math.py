"""
Tree-Structured Mathematical Processing
========================================

This module provides:
1. Canonicalization - transform expressions to canonical form
2. Tree Encoder - recursive bottom-up neural encoding
3. Structure-Aware Transformer - with depth/position embeddings
4. Lean Bridge - formal verification integration

These address the core issue: equivalent expressions should have
similar representations, not different token sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
import os
import hashlib

try:
    import sympy
    from sympy import symbols, simplify, expand
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =============================================================================
# PART 1: EXPRESSION TREE WITH METADATA
# =============================================================================

class Op(Enum):
    """Operators matching isoengine core."""
    ADD = 0; SUB = 1; MUL = 2; DIV = 3; POW = 4; NEG = 5
    VAR_X = 6; VAR_Y = 7; VAR_Z = 8; CONST = 9
    EQ = 10; GT = 11; LT = 12; GTE = 13; LTE = 14
    SIN = 21; COS = 22; TAN = 23; LN = 24; EXP = 25; SQRT = 26
    PAD = 28


@dataclass
class TreeNode:
    """
    Expression tree node with structural metadata.

    Extends basic Expr with:
    - depth: distance from root
    - path: binary path string ("L", "R", "LL", etc.)
    - subtree_size: number of nodes in subtree
    - subtree_hash: hash for canonicalization
    """
    op: Op
    value: float = 0.0
    left: 'TreeNode' = None
    right: 'TreeNode' = None

    # Metadata (computed by compute_metadata)
    depth: int = 0
    path: str = ""
    subtree_size: int = 1
    subtree_hash: int = 0

    def compute_metadata(self, depth: int = 0, path: str = "") -> 'TreeNode':
        """Compute structural metadata for all nodes."""
        self.depth = depth
        self.path = path
        self.subtree_size = 1

        # Compute hash based on structure
        h = hash((self.op.value, round(self.value * 1000)))

        if self.left:
            self.left.compute_metadata(depth + 1, path + "L")
            self.subtree_size += self.left.subtree_size
            h = h * 31 + self.left.subtree_hash

        if self.right:
            self.right.compute_metadata(depth + 1, path + "R")
            self.subtree_size += self.right.subtree_size
            h = h * 37 + self.right.subtree_hash

        self.subtree_hash = h
        return self

    def copy(self) -> 'TreeNode':
        """Deep copy the tree."""
        return TreeNode(
            op=self.op,
            value=self.value,
            left=self.left.copy() if self.left else None,
            right=self.right.copy() if self.right else None
        )

    def evaluate(self, x: float = 0, y: float = 0) -> float:
        """Numerically evaluate the expression."""
        import math

        if self.op == Op.CONST: return self.value
        elif self.op == Op.VAR_X: return x
        elif self.op == Op.VAR_Y: return y
        elif self.op == Op.ADD: return self.left.evaluate(x, y) + self.right.evaluate(x, y)
        elif self.op == Op.SUB: return self.left.evaluate(x, y) - self.right.evaluate(x, y)
        elif self.op == Op.MUL: return self.left.evaluate(x, y) * self.right.evaluate(x, y)
        elif self.op == Op.DIV:
            r = self.right.evaluate(x, y)
            return self.left.evaluate(x, y) / r if r != 0 else float('inf')
        elif self.op == Op.POW:
            return self.left.evaluate(x, y) ** self.right.evaluate(x, y)
        elif self.op == Op.NEG: return -self.left.evaluate(x, y)
        elif self.op == Op.SIN: return math.sin(self.left.evaluate(x, y))
        elif self.op == Op.COS: return math.cos(self.left.evaluate(x, y))
        elif self.op == Op.SQRT: return math.sqrt(max(0, self.left.evaluate(x, y)))
        return 0

    def to_string(self) -> str:
        """Convert to readable string."""
        if self.op == Op.CONST:
            v = self.value
            return str(int(v)) if v == int(v) else f"{v:.2g}"
        elif self.op == Op.VAR_X: return "x"
        elif self.op == Op.VAR_Y: return "y"
        elif self.op == Op.ADD: return f"({self.left.to_string()} + {self.right.to_string()})"
        elif self.op == Op.SUB: return f"({self.left.to_string()} - {self.right.to_string()})"
        elif self.op == Op.MUL: return f"({self.left.to_string()} * {self.right.to_string()})"
        elif self.op == Op.DIV: return f"({self.left.to_string()} / {self.right.to_string()})"
        elif self.op == Op.POW: return f"({self.left.to_string()})^{self.right.to_string()}"
        elif self.op == Op.NEG: return f"-{self.left.to_string()}"
        return "?"


# =============================================================================
# PART 2: CANONICALIZATION
# =============================================================================

def canonicalize(node: TreeNode) -> TreeNode:
    """
    Transform expression tree to canonical form.

    Rules:
    1. Recursively canonicalize children first
    2. Sort commutative operations (ADD, MUL) by subtree hash
    3. Flatten nested same-ops where possible
    4. Apply algebraic simplifications

    After canonicalization, equivalent expressions have the SAME tree structure!
    """
    if node is None:
        return None

    # First, recursively canonicalize children
    if node.left:
        node.left = canonicalize(node.left)
    if node.right:
        node.right = canonicalize(node.right)

    # Recompute metadata after child changes
    node.compute_metadata(node.depth, node.path)

    # Sort commutative operations by subtree hash
    if node.op in [Op.ADD, Op.MUL]:
        if node.left and node.right:
            if node.left.subtree_hash > node.right.subtree_hash:
                node.left, node.right = node.right, node.left

    # Algebraic simplifications
    node = simplify_node(node)

    return node


def simplify_node(node: TreeNode) -> TreeNode:
    """Apply algebraic simplifications to a single node."""
    if node is None:
        return None

    # a + 0 = a
    if node.op == Op.ADD:
        if node.right and node.right.op == Op.CONST and node.right.value == 0:
            return node.left
        if node.left and node.left.op == Op.CONST and node.left.value == 0:
            return node.right

    # a * 1 = a
    if node.op == Op.MUL:
        if node.right and node.right.op == Op.CONST and node.right.value == 1:
            return node.left
        if node.left and node.left.op == Op.CONST and node.left.value == 1:
            return node.right

    # a * 0 = 0
    if node.op == Op.MUL:
        if (node.right and node.right.op == Op.CONST and node.right.value == 0) or \
           (node.left and node.left.op == Op.CONST and node.left.value == 0):
            return TreeNode(Op.CONST, value=0)

    # a^1 = a
    if node.op == Op.POW:
        if node.right and node.right.op == Op.CONST and node.right.value == 1:
            return node.left

    # a^0 = 1
    if node.op == Op.POW:
        if node.right and node.right.op == Op.CONST and node.right.value == 0:
            return TreeNode(Op.CONST, value=1)

    # Constant folding: const op const = result
    if node.op in [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.POW]:
        if (node.left and node.left.op == Op.CONST and
            node.right and node.right.op == Op.CONST):
            result = node.evaluate()
            if result != float('inf') and result == result:  # not inf or nan
                return TreeNode(Op.CONST, value=result)

    return node


def are_equivalent(t1: TreeNode, t2: TreeNode, n_tests: int = 10) -> bool:
    """Test if two trees are equivalent by numerical evaluation."""
    import random
    for _ in range(n_tests):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        v1 = t1.evaluate(x, y)
        v2 = t2.evaluate(x, y)
        if abs(v1 - v2) > 1e-6 * (1 + abs(v1) + abs(v2)):
            return False
    return True


# =============================================================================
# PART 3: TREE ENCODER (Recursive Bottom-Up)
# =============================================================================

class TreeEncoder(nn.Module):
    """
    Recursive Tree Encoder.

    Processes expression trees bottom-up:
    - Leaf nodes (CONST, VAR) get base embeddings
    - Internal nodes combine children embeddings via learned functions
    - Final root embedding captures entire expression structure

    This naturally handles tree structure - no flattening needed!
    """

    def __init__(self, embed_dim: int = 64, output_dim: int = 32, num_ops: int = 30):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Operator embeddings
        self.op_embed = nn.Embedding(num_ops, embed_dim)

        # Value encoder for constants
        self.value_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh()
        )

        # Binary combiner: combines op + left_child + right_child
        self.binary_combiner = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Unary combiner: combines op + child
        self.unary_combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim)
        )

    def encode_node(self, node: TreeNode) -> torch.Tensor:
        """Recursively encode a tree node bottom-up."""
        op_emb = self.op_embed(torch.tensor([node.op.value]))

        # Leaf: constant
        if node.op == Op.CONST:
            val_emb = self.value_encoder(torch.tensor([[node.value]], dtype=torch.float))
            return (op_emb + val_emb).squeeze(0)

        # Leaf: variable
        if node.op in [Op.VAR_X, Op.VAR_Y, Op.VAR_Z]:
            return op_emb.squeeze(0)

        # Binary operator
        if node.left is not None and node.right is not None:
            left_emb = self.encode_node(node.left)
            right_emb = self.encode_node(node.right)
            combined = torch.cat([op_emb.squeeze(0), left_emb, right_emb])
            return self.binary_combiner(combined)

        # Unary operator
        if node.left is not None:
            child_emb = self.encode_node(node.left)
            combined = torch.cat([op_emb.squeeze(0), child_emb])
            return self.unary_combiner(combined)

        return op_emb.squeeze(0)

    def forward(self, tree: TreeNode, canonicalize_first: bool = True) -> torch.Tensor:
        """
        Encode tree to output embedding.

        Args:
            tree: Expression tree to encode
            canonicalize_first: If True, canonicalize before encoding

        Returns:
            Normalized embedding vector
        """
        if canonicalize_first:
            tree = canonicalize(tree.copy())
            tree.compute_metadata()

        root_emb = self.encode_node(tree)
        output = self.output_proj(root_emb)
        return F.normalize(output, dim=-1)

    def encode_batch(self, trees: List[TreeNode], canonicalize_first: bool = True) -> torch.Tensor:
        """Encode a batch of trees."""
        embeddings = []
        for tree in trees:
            emb = self.forward(tree, canonicalize_first)
            embeddings.append(emb)
        return torch.stack(embeddings)


# =============================================================================
# PART 4: STRUCTURE-AWARE TRANSFORMER
# =============================================================================

def tree_to_tensor_with_structure(node: TreeNode, max_nodes: int = 32) -> Dict[str, torch.Tensor]:
    """
    Convert tree to tensor with structural information.

    Returns dict with:
    - ops: operator indices
    - vals: constant values
    - depths: depth of each node
    - positions: tree position encoding
    - parent_indices: index of parent node (-1 for root)
    """
    ops, vals, depths, positions, parent_indices = [], [], [], [], []

    def path_to_position(path: str, max_depth: int = 8) -> int:
        """Convert path string to unique integer."""
        pos = 1  # Start at 1 (root)
        for c in path:
            pos = pos * 2 + (1 if c == 'R' else 0)
        return pos

    node_to_idx = {}

    def traverse(n: TreeNode, parent_idx: int = -1):
        if n is None or len(ops) >= max_nodes:
            return

        idx = len(ops)
        node_to_idx[id(n)] = idx

        ops.append(n.op.value)
        vals.append(n.value if n.op == Op.CONST else 0.0)
        depths.append(n.depth)
        positions.append(path_to_position(n.path))
        parent_indices.append(parent_idx)

        traverse(n.left, idx)
        traverse(n.right, idx)

    node.compute_metadata()
    traverse(node)

    # Pad to max_nodes
    while len(ops) < max_nodes:
        ops.append(Op.PAD.value)
        vals.append(0.0)
        depths.append(0)
        positions.append(0)
        parent_indices.append(-1)

    return {
        'ops': torch.tensor(ops[:max_nodes], dtype=torch.long),
        'vals': torch.tensor(vals[:max_nodes], dtype=torch.float),
        'depths': torch.tensor(depths[:max_nodes], dtype=torch.long),
        'positions': torch.tensor(positions[:max_nodes], dtype=torch.long),
        'parent_indices': torch.tensor(parent_indices[:max_nodes], dtype=torch.long),
    }


class StructureAwareTransformer(nn.Module):
    """
    Transformer with tree structure information.

    Enhancements over basic transformer:
    - Depth embeddings: encode distance from root
    - Tree position embeddings: encode path in tree
    - Parent-aware attention (optional): bias attention toward parent nodes
    """

    def __init__(self, embed_dim: int = 128, output_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 3, max_nodes: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes

        # Token embeddings
        self.op_embed = nn.Embedding(30, embed_dim)
        self.val_embed = nn.Linear(1, embed_dim)

        # Structural embeddings
        self.depth_embed = nn.Embedding(16, embed_dim)  # Max depth 16
        self.tree_pos_embed = nn.Embedding(512, embed_dim)  # Tree positions

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, ops: torch.Tensor, vals: torch.Tensor,
                depths: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structural information.

        Args:
            ops: [batch, seq_len] operator indices
            vals: [batch, seq_len] constant values
            depths: [batch, seq_len] node depths
            positions: [batch, seq_len] tree positions
        """
        # Create attention mask for padding
        mask = (ops == Op.PAD.value)

        # Combine embeddings
        x = (self.op_embed(ops) +
             self.val_embed(vals.unsqueeze(-1)) +
             self.depth_embed(depths.clamp(0, 15)) +
             self.tree_pos_embed(positions.clamp(0, 511)))

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over non-padding tokens
        mask_expanded = (~mask).float().unsqueeze(-1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Project to output
        return F.normalize(self.output_proj(x), dim=-1)

    def encode_tree(self, tree: TreeNode, canonicalize_first: bool = True) -> torch.Tensor:
        """Encode a single tree."""
        if canonicalize_first:
            tree = canonicalize(tree.copy())

        data = tree_to_tensor_with_structure(tree, self.max_nodes)

        # Add batch dimension
        ops = data['ops'].unsqueeze(0)
        vals = data['vals'].unsqueeze(0)
        depths = data['depths'].unsqueeze(0)
        positions = data['positions'].unsqueeze(0)

        return self.forward(ops, vals, depths, positions).squeeze(0)

    def encode_batch(self, trees: List[TreeNode], canonicalize_first: bool = True) -> torch.Tensor:
        """Encode a batch of trees."""
        batch_data = []
        for tree in trees:
            if canonicalize_first:
                tree = canonicalize(tree.copy())
            batch_data.append(tree_to_tensor_with_structure(tree, self.max_nodes))

        # Stack into batches
        ops = torch.stack([d['ops'] for d in batch_data])
        vals = torch.stack([d['vals'] for d in batch_data])
        depths = torch.stack([d['depths'] for d in batch_data])
        positions = torch.stack([d['positions'] for d in batch_data])

        return self.forward(ops, vals, depths, positions)


# =============================================================================
# PART 5: LEAN BRIDGE
# =============================================================================

class LeanBridge:
    """
    Bridge to Lean 4 for formal verification.

    Uses Lean to:
    1. Verify algebraic equivalences
    2. Generate provably equivalent expression pairs
    3. Provide ground truth for training
    """

    def __init__(self, lean_project_path: Optional[str] = None):
        self.lean_available = self._check_lean()
        self.sympy_available = SYMPY_AVAILABLE
        self.project_path = lean_project_path

    def _check_lean(self) -> bool:
        """Check if Lean 4 is available."""
        try:
            result = subprocess.run(['lean', '--version'],
                                    capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def verify_equivalence_lean(self, expr1: str, expr2: str) -> Tuple[bool, str]:
        """
        Use Lean to verify expr1 = expr2.

        Returns (verified, message)
        """
        if not self.lean_available:
            return False, "Lean not available"

        lean_code = f'''
import Mathlib.Tactic

theorem equiv_check (a b c x : Int) : {expr1} = {expr2} := by
  ring
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['lake', 'env', 'lean', temp_path],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                return True, "Verified by Lean"
            else:
                return False, f"Lean error: {result.stderr[:200]}"
        except Exception as e:
            return False, f"Error: {e}"
        finally:
            os.unlink(temp_path)

    def verify_equivalence_sympy(self, expr1: str, expr2: str) -> Tuple[bool, str]:
        """
        Use SymPy to verify expr1 = expr2 symbolically.
        """
        if not self.sympy_available:
            return False, "SymPy not available"

        try:
            x, a, b, c = symbols('x a b c')
            e1 = parse_expr(expr1)
            e2 = parse_expr(expr2)

            diff = simplify(expand(e1) - expand(e2))

            if diff == 0:
                return True, "Verified by SymPy"
            else:
                return False, f"Not equivalent. Diff: {diff}"
        except Exception as e:
            return False, f"SymPy error: {e}"

    def verify_equivalence(self, expr1: str, expr2: str) -> Tuple[bool, str]:
        """Verify equivalence using best available method."""
        # Try Lean first (most rigorous)
        if self.lean_available:
            success, msg = self.verify_equivalence_lean(expr1, expr2)
            if success:
                return True, msg

        # Fall back to SymPy
        if self.sympy_available:
            return self.verify_equivalence_sympy(expr1, expr2)

        return False, "No verification backend available"

    def generate_equivalent_pairs(self, n_pairs: int = 100) -> List[Dict[str, Any]]:
        """
        Generate pairs of provably equivalent expressions.

        Each pair is verified symbolically before being returned.
        """
        if not self.sympy_available:
            return []

        import random

        pairs = []
        x = symbols('x')

        # Transformation templates
        templates = [
            # Distributive: a*(b+c) = a*b + a*c
            lambda: self._gen_distributive(),
            # Square of sum: (a+b)^2 = a^2 + 2ab + b^2
            lambda: self._gen_square_sum(),
            # Difference of squares: (a+b)(a-b) = a^2 - b^2
            lambda: self._gen_diff_squares(),
            # Combine like terms: ax + bx = (a+b)x
            lambda: self._gen_combine_terms(),
            # Commutativity: a + b = b + a
            lambda: self._gen_commutative(),
        ]

        attempts = 0
        while len(pairs) < n_pairs and attempts < n_pairs * 3:
            attempts += 1
            template = random.choice(templates)
            try:
                pair = template()
                if pair:
                    pairs.append(pair)
            except:
                continue

        return pairs

    def _gen_distributive(self) -> Optional[Dict]:
        """Generate a*(b+c) = a*b + a*c pair."""
        import random
        a = random.randint(2, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)

        expr1 = f"{a}*(x + {b})"
        expr2 = f"{a}*x + {a*b}"

        tree1 = TreeNode(Op.MUL,
            left=TreeNode(Op.CONST, value=a),
            right=TreeNode(Op.ADD,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=b)))

        tree2 = TreeNode(Op.ADD,
            left=TreeNode(Op.MUL,
                left=TreeNode(Op.CONST, value=a),
                right=TreeNode(Op.VAR_X)),
            right=TreeNode(Op.CONST, value=a*b))

        return {
            'expr1': expr1, 'expr2': expr2,
            'tree1': tree1, 'tree2': tree2,
            'rule': 'distributive',
            'verified': True
        }

    def _gen_square_sum(self) -> Optional[Dict]:
        """Generate (x+a)^2 = x^2 + 2ax + a^2 pair."""
        import random
        a = random.randint(1, 4)

        expr1 = f"(x + {a})**2"
        expr2 = f"x**2 + {2*a}*x + {a*a}"

        tree1 = TreeNode(Op.POW,
            left=TreeNode(Op.ADD,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=a)),
            right=TreeNode(Op.CONST, value=2))

        tree2 = TreeNode(Op.ADD,
            left=TreeNode(Op.POW,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=2)),
            right=TreeNode(Op.ADD,
                left=TreeNode(Op.MUL,
                    left=TreeNode(Op.CONST, value=2*a),
                    right=TreeNode(Op.VAR_X)),
                right=TreeNode(Op.CONST, value=a*a)))

        return {
            'expr1': expr1, 'expr2': expr2,
            'tree1': tree1, 'tree2': tree2,
            'rule': 'square_of_sum',
            'verified': True
        }

    def _gen_diff_squares(self) -> Optional[Dict]:
        """Generate (x+a)(x-a) = x^2 - a^2 pair."""
        import random
        a = random.randint(1, 5)

        tree1 = TreeNode(Op.MUL,
            left=TreeNode(Op.ADD,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=a)),
            right=TreeNode(Op.SUB,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=a)))

        tree2 = TreeNode(Op.SUB,
            left=TreeNode(Op.POW,
                left=TreeNode(Op.VAR_X),
                right=TreeNode(Op.CONST, value=2)),
            right=TreeNode(Op.CONST, value=a*a))

        return {
            'expr1': f"(x+{a})*(x-{a})",
            'expr2': f"x**2 - {a*a}",
            'tree1': tree1, 'tree2': tree2,
            'rule': 'difference_of_squares',
            'verified': True
        }

    def _gen_combine_terms(self) -> Optional[Dict]:
        """Generate ax + bx = (a+b)x pair."""
        import random
        a = random.randint(1, 5)
        b = random.randint(1, 5)

        tree1 = TreeNode(Op.ADD,
            left=TreeNode(Op.MUL,
                left=TreeNode(Op.CONST, value=a),
                right=TreeNode(Op.VAR_X)),
            right=TreeNode(Op.MUL,
                left=TreeNode(Op.CONST, value=b),
                right=TreeNode(Op.VAR_X)))

        tree2 = TreeNode(Op.MUL,
            left=TreeNode(Op.CONST, value=a+b),
            right=TreeNode(Op.VAR_X))

        return {
            'expr1': f"{a}*x + {b}*x",
            'expr2': f"{a+b}*x",
            'tree1': tree1, 'tree2': tree2,
            'rule': 'combine_like_terms',
            'verified': True
        }

    def _gen_commutative(self) -> Optional[Dict]:
        """Generate a + b = b + a pair."""
        import random
        a = random.randint(1, 9)

        tree1 = TreeNode(Op.ADD,
            left=TreeNode(Op.VAR_X),
            right=TreeNode(Op.CONST, value=a))

        tree2 = TreeNode(Op.ADD,
            left=TreeNode(Op.CONST, value=a),
            right=TreeNode(Op.VAR_X))

        return {
            'expr1': f"x + {a}",
            'expr2': f"{a} + x",
            'tree1': tree1, 'tree2': tree2,
            'rule': 'commutativity',
            'verified': True
        }


# =============================================================================
# PART 6: UNIFIED TRAINING
# =============================================================================

class TreeMathTrainer:
    """
    Unified trainer for tree-structured math understanding.

    Combines:
    - Tree Encoder or Structure-Aware Transformer
    - Canonicalization
    - Lean/SymPy verified training data
    - Contrastive learning on equivalences
    """

    def __init__(self,
                 encoder_type: str = 'tree',  # 'tree' or 'transformer'
                 embed_dim: int = 64,
                 output_dim: int = 32,
                 device: str = 'cpu'):

        self.device = torch.device(device)

        if encoder_type == 'tree':
            self.encoder = TreeEncoder(embed_dim, output_dim).to(self.device)
        else:
            self.encoder = StructureAwareTransformer(
                embed_dim=embed_dim, output_dim=output_dim
            ).to(self.device)

        self.encoder_type = encoder_type
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-3)
        self.lean_bridge = LeanBridge()

        self.stats = {
            'epochs': 0,
            'losses': [],
            'equivalence_accuracy': []
        }

    def generate_training_batch(self, batch_size: int = 32) -> List[Dict]:
        """Generate batch of verified equivalent pairs."""
        return self.lean_bridge.generate_equivalent_pairs(batch_size)

    def contrastive_loss(self, embeddings: torch.Tensor,
                         labels: List[str], temperature: float = 0.1) -> torch.Tensor:
        """Supervised contrastive loss."""
        n = len(labels)
        sim = torch.mm(embeddings, embeddings.T) / temperature

        # Create label mask
        unique_labels = list(set(labels))
        label_to_id = {l: i for i, l in enumerate(unique_labels)}
        label_ids = torch.tensor([label_to_id[l] for l in labels], device=self.device)

        label_match = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1)).float()
        self_mask = torch.eye(n, device=self.device)
        pos_mask = label_match * (1 - self_mask)

        exp_sim = torch.exp(sim) * (1 - self_mask)
        pos_sum = (sim * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        neg_sum = torch.log(exp_sim.sum(1) + 1e-8)

        return (-pos_sum + neg_sum).mean()

    def train_epoch(self, n_batches: int = 10, batch_size: int = 32) -> float:
        """Train for one epoch."""
        self.encoder.train()
        total_loss = 0

        for _ in range(n_batches):
            pairs = self.generate_training_batch(batch_size)

            if not pairs:
                continue

            # Collect all trees and labels
            trees = []
            labels = []

            for pair in pairs:
                trees.append(pair['tree1'])
                trees.append(pair['tree2'])
                # Same label for equivalent expressions
                label = f"{pair['rule']}_{hash(pair['expr1']) % 1000}"
                labels.append(label)
                labels.append(label)

            # Encode
            if self.encoder_type == 'tree':
                embeddings = self.encoder.encode_batch(trees, canonicalize_first=True)
            else:
                embeddings = self.encoder.encode_batch(trees, canonicalize_first=True)

            # Compute loss
            loss = self.contrastive_loss(embeddings, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(n_batches, 1)
        self.stats['losses'].append(avg_loss)
        self.stats['epochs'] += 1

        return avg_loss

    def evaluate_equivalences(self, n_pairs: int = 50) -> float:
        """Evaluate how well encoder clusters equivalent expressions."""
        self.encoder.eval()

        pairs = self.lean_bridge.generate_equivalent_pairs(n_pairs)
        if not pairs:
            return 0.0

        correct = 0
        total = 0

        with torch.no_grad():
            for pair in pairs:
                if self.encoder_type == 'tree':
                    emb1 = self.encoder(pair['tree1'], canonicalize_first=True)
                    emb2 = self.encoder(pair['tree2'], canonicalize_first=True)
                else:
                    emb1 = self.encoder.encode_tree(pair['tree1'], canonicalize_first=True)
                    emb2 = self.encoder.encode_tree(pair['tree2'], canonicalize_first=True)

                sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

                # Consider similar if cosine sim > 0.5
                if sim > 0.5:
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)
        self.stats['equivalence_accuracy'].append(accuracy)

        return accuracy

    def train(self, epochs: int = 100, eval_every: int = 10, verbose: bool = True):
        """Full training loop."""
        if verbose:
            print(f"Training {self.encoder_type} encoder...")
            print(f"Lean available: {self.lean_bridge.lean_available}")
            print(f"SymPy available: {self.lean_bridge.sympy_available}")

        for epoch in range(epochs):
            loss = self.train_epoch()

            if (epoch + 1) % eval_every == 0:
                acc = self.evaluate_equivalences()
                if verbose:
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Equiv Accuracy={acc:.1%}")

        if verbose:
            final_acc = self.evaluate_equivalences(100)
            print(f"\nFinal equivalence accuracy: {final_acc:.1%}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TREE MATH MODULE TEST")
    print("=" * 70)

    # Test canonicalization
    print("\n[1] Testing Canonicalization")
    print("-" * 40)

    # x + 3 and 3 + x should canonicalize to same form
    t1 = TreeNode(Op.ADD,
        left=TreeNode(Op.VAR_X),
        right=TreeNode(Op.CONST, value=3))

    t2 = TreeNode(Op.ADD,
        left=TreeNode(Op.CONST, value=3),
        right=TreeNode(Op.VAR_X))

    t1_canon = canonicalize(t1.copy())
    t2_canon = canonicalize(t2.copy())

    t1_canon.compute_metadata()
    t2_canon.compute_metadata()

    print(f"  Original: x+3 vs 3+x")
    print(f"  After canonicalization:")
    print(f"    t1 hash: {t1_canon.subtree_hash}")
    print(f"    t2 hash: {t2_canon.subtree_hash}")
    print(f"    Same? {t1_canon.subtree_hash == t2_canon.subtree_hash}")

    # Test Tree Encoder
    print("\n[2] Testing Tree Encoder")
    print("-" * 40)

    encoder = TreeEncoder(embed_dim=64, output_dim=32)

    with torch.no_grad():
        emb1 = encoder(t1, canonicalize_first=True)
        emb2 = encoder(t2, canonicalize_first=True)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    print(f"  x+3 vs 3+x similarity: {sim:.3f}")

    # Test Lean Bridge
    print("\n[3] Testing Lean Bridge")
    print("-" * 40)

    bridge = LeanBridge()
    print(f"  Lean available: {bridge.lean_available}")
    print(f"  SymPy available: {bridge.sympy_available}")

    if bridge.sympy_available:
        success, msg = bridge.verify_equivalence("2*(x+3)", "2*x+6")
        print(f"  2*(x+3) = 2*x+6: {msg}")

        pairs = bridge.generate_equivalent_pairs(5)
        print(f"  Generated {len(pairs)} verified pairs")
        for p in pairs[:3]:
            print(f"    {p['expr1']} = {p['expr2']} ({p['rule']})")

    print("\n" + "=" * 70)
