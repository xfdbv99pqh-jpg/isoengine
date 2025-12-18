"""
Neural encoder components: HyperbolicEncoder, ContrastiveTrainer, SolutionPredictor
"""

import random
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .core import (
    Op, Expr,
    Const, VarX, VarY, Add, Mul, Pow, Eq, Gt, Lt, System,
    ProblemGenerator
)


if TORCH_AVAILABLE:
    
    class HyperbolicEncoder(nn.Module):
        """Encode expressions into hyperbolic space."""
        
        def __init__(self, embed_dim=256, hyperbolic_dim=64, num_heads=8, num_layers=4, max_nodes=32):
            super().__init__()
            self.max_nodes = max_nodes
            self.hyperbolic_dim = hyperbolic_dim
            self.embed_dim = embed_dim
            
            self.op_embed = nn.Embedding(len(Op), embed_dim)
            self.val_embed = nn.Linear(1, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, max_nodes, embed_dim) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=0.1,
                batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.to_hyperbolic = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hyperbolic_dim)
            )
            self.norm = nn.LayerNorm(embed_dim)
        
        def forward(self, op_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            mask = (op_ids == Op.PAD.value)
            x = self.op_embed(op_ids) + self.val_embed(values.unsqueeze(-1))
            x = x + self.pos_embed[:, :op_ids.shape[1], :]
            x = self.transformer(x, src_key_padding_mask=mask)
            
            m = (~mask).float().unsqueeze(-1)
            x = (x * m).sum(1) / (m.sum(1) + 1e-8)
            x = self.norm(x)
            
            h = self.to_hyperbolic(x)
            h = F.normalize(h, p=2, dim=-1)
            return h
        
        def encode_exprs(self, exprs: List[Expr], device=None) -> torch.Tensor:
            if device is None:
                device = next(self.parameters()).device
            
            ops, vals = [], []
            for e in exprs:
                o, v = e.to_tensor(self.max_nodes)
                ops.append(o)
                vals.append(v)
            
            ops = torch.stack(ops).to(device)
            vals = torch.stack(vals).to(device)
            return self.forward(ops, vals)
        
        def similarity(self, expr1: Expr, expr2: Expr) -> float:
            with torch.no_grad():
                emb1 = self.encode_exprs([expr1])
                emb2 = self.encode_exprs([expr2])
                return F.cosine_similarity(emb1, emb2).item()
    
    
    class ContrastiveTrainer:
        """Train encoder with supervised contrastive loss."""
        
        def __init__(self, encoder: HyperbolicEncoder, device=None):
            self.encoder = encoder
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder.to(self.device)
            self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=0.01)
            self.scheduler = None
            self.temperature = 0.1
        
        def generate_batch(self, samples_per_class: int = 3) -> Tuple[List[Expr], List]:
            exprs, labels = [], []
            
            # Linear by solution
            for sol in range(-5, 6):
                for _ in range(samples_per_class):
                    expr, _ = ProblemGenerator.linear_equation(sol)
                    exprs.append(expr)
                    labels.append(('linear', sol))
            
            # Quadratic by roots
            for r1, r2 in [(1,2), (1,3), (-1,1), (-2,2), (0,1), (0,2)]:
                for _ in range(samples_per_class):
                    expr, _ = ProblemGenerator.quadratic_equation(r1, r2)
                    exprs.append(expr)
                    labels.append(('quadratic', (r1, r2)))
            
            # Systems
            for x, y in [(1,1), (1,2), (2,1), (0,1), (-1,1)]:
                for _ in range(samples_per_class):
                    expr, _ = ProblemGenerator.system_equation(x, y)
                    exprs.append(expr)
                    labels.append(('system', (x, y)))
            
            # Inequalities
            for boundary in [-3, -1, 0, 1, 2, 3]:
                for direction in ['gt', 'lt']:
                    for _ in range(samples_per_class):
                        expr, _ = ProblemGenerator.linear_inequality(boundary, direction)
                        exprs.append(expr)
                        labels.append(('ineq', (boundary, direction)))
            
            return exprs, labels
        
        def supcon_loss(self, features: torch.Tensor, labels: List) -> torch.Tensor:
            unique = list(set(labels))
            label_to_id = {l: i for i, l in enumerate(unique)}
            label_ids = torch.tensor([label_to_id[l] for l in labels]).to(self.device)
            
            batch_size = features.shape[0]
            sim = torch.matmul(features, features.T) / self.temperature
            
            labels_col = label_ids.view(-1, 1)
            mask = torch.eq(labels_col, labels_col.T).float()
            self_mask = torch.eye(batch_size).to(self.device)
            mask = mask - self_mask
            
            logits_max, _ = torch.max(sim, dim=1, keepdim=True)
            logits = sim - logits_max.detach()
            exp_logits = torch.exp(logits) * (1 - self_mask)
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
            
            mask_sum = mask.sum(dim=1).clamp(min=1)
            mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
            return -mean_log_prob.mean()
        
        def train(self, epochs: int = 2000, verbose: bool = True):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
            
            for epoch in range(epochs):
                self.encoder.train()
                exprs, labels = self.generate_batch()
                features = self.encoder.encode_exprs(exprs, self.device)
                loss = self.supcon_loss(features, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                if verbose and epoch % 500 == 0:
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}")
            
            if verbose:
                print("Training complete.")
    
    
    class SolutionPredictor(nn.Module):
        """Predict numeric solutions from embeddings."""
        
        def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, output_dim: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            return self.net(embeddings)

else:
    # Stubs if torch not available
    class HyperbolicEncoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for HyperbolicEncoder")
    
    class ContrastiveTrainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for ContrastiveTrainer")
    
    class SolutionPredictor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for SolutionPredictor")
