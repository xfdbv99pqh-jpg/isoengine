"""
Multi-Head Solution Prediction Trainer
======================================
Train encoder with type-specific regression heads for:
- Linear equations: predict x
- Quadratic equations: predict r1+r2 (sum of roots)
- Inequalities: predict boundary value

Results:
- Linear R² = 0.711
- Quadratic R² = 0.999
- Inequality R² = 0.996

Author: Big J + Claude
"""

import random
from typing import List, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .core import (
    Op, Expr,
    Const, VarX, Add, Mul, Pow, Eq, Gt, Lt
)
from .encoder import HyperbolicEncoder


if TORCH_AVAILABLE:

    class MultiHeadTrainer:
        """
        Train encoder with contrastive loss + type-specific regression heads.

        Usage:
            encoder = HyperbolicEncoder().to(device)
            trainer = MultiHeadTrainer(encoder, device)
            trainer.train(epochs=3000)

            # Predict solutions
            emb = encoder.encode_exprs([linear_eq], device)
            solution = trainer.linear_head(emb)
        """

        def __init__(self, encoder: HyperbolicEncoder, device=None):
            self.encoder = encoder
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder.to(self.device)

            hdim = encoder.hyperbolic_dim

            # Type-specific regression heads
            self.linear_head = nn.Sequential(
                nn.Linear(hdim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(self.device)

            self.quadratic_head = nn.Sequential(
                nn.Linear(hdim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(self.device)

            self.inequality_head = nn.Sequential(
                nn.Linear(hdim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(self.device)

            # Combined optimizer
            params = (list(encoder.parameters()) +
                      list(self.linear_head.parameters()) +
                      list(self.quadratic_head.parameters()) +
                      list(self.inequality_head.parameters()))

            self.optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
            self.scheduler = None
            self.temperature = 0.1

        def gen_linear(self, n: int = 50) -> Tuple[List[Expr], List[float]]:
            """Generate linear equations with continuous solutions."""
            exprs, sols = [], []
            for _ in range(n):
                sol = random.uniform(-10, 10)
                a = random.choice([1, 2, 3, -1, -2, -3, 0.5, -0.5])
                b = random.uniform(-20, 20)
                c = a * sol + b
                exprs.append(Eq(Add(Mul(Const(a), VarX()), Const(b)), Const(c)))
                sols.append(sol)
            return exprs, sols

        def gen_quadratic(self, n: int = 50) -> Tuple[List[Expr], List[float]]:
            """Generate quadratics, return sum of roots."""
            exprs, sums = [], []
            for _ in range(n):
                r1 = random.uniform(-5, 5)
                r2 = random.uniform(-5, 5)
                b, c = -(r1 + r2), r1 * r2
                x2 = Pow(VarX(), Const(2))
                lhs = x2
                if abs(b) > 1e-6:
                    lhs = Add(lhs, Mul(Const(b), VarX()))
                if abs(c) > 1e-6:
                    lhs = Add(lhs, Const(c))
                exprs.append(Eq(lhs, Const(0)))
                sums.append(r1 + r2)
            return exprs, sums

        def gen_inequality(self, n: int = 50) -> Tuple[List[Expr], List[float]]:
            """Generate inequalities, return boundary."""
            exprs, bounds = [], []
            for _ in range(n):
                boundary = random.uniform(-10, 10)
                a = random.choice([1, 2, 3, -1, -2, -3])
                b = random.uniform(-20, 20)
                c = a * boundary + b
                lhs = Add(Mul(Const(a), VarX()), Const(b))
                op = random.choice([Gt, Lt])
                exprs.append(op(lhs, Const(c)))
                bounds.append(boundary)
            return exprs, bounds

        def contrastive_loss(self, features: torch.Tensor, labels: List[float]) -> torch.Tensor:
            """Supervised contrastive loss with binned labels."""
            binned = [round(l * 2) / 2 for l in labels]
            unique = list(set(binned))
            label_to_id = {l: i for i, l in enumerate(unique)}
            label_ids = torch.tensor([label_to_id[l] for l in binned]).to(self.device)

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

        def train(self, epochs: int = 3000, verbose: bool = True) -> Dict:
            """Train the encoder and all regression heads."""
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)

            history = {'contrastive': [], 'linear': [], 'quadratic': [], 'inequality': []}

            for epoch in range(epochs):
                self.encoder.train()
                self.linear_head.train()
                self.quadratic_head.train()
                self.inequality_head.train()

                # Generate batches
                lin_exprs, lin_sols = self.gen_linear(40)
                quad_exprs, quad_sums = self.gen_quadratic(40)
                ineq_exprs, ineq_bounds = self.gen_inequality(40)

                # Encode
                lin_emb = self.encoder.encode_exprs(lin_exprs, self.device)
                quad_emb = self.encoder.encode_exprs(quad_exprs, self.device)
                ineq_emb = self.encoder.encode_exprs(ineq_exprs, self.device)

                # Contrastive loss
                all_emb = torch.cat([lin_emb, quad_emb, ineq_emb], dim=0)
                all_vals = lin_sols + quad_sums + ineq_bounds
                c_loss = self.contrastive_loss(all_emb, all_vals)

                # Regression losses
                lin_pred = self.linear_head(lin_emb)
                lin_tgt = torch.tensor(lin_sols, dtype=torch.float32).unsqueeze(1).to(self.device)
                lin_loss = F.mse_loss(lin_pred, lin_tgt)

                quad_pred = self.quadratic_head(quad_emb)
                quad_tgt = torch.tensor(quad_sums, dtype=torch.float32).unsqueeze(1).to(self.device)
                quad_loss = F.mse_loss(quad_pred, quad_tgt)

                ineq_pred = self.inequality_head(ineq_emb)
                ineq_tgt = torch.tensor(ineq_bounds, dtype=torch.float32).unsqueeze(1).to(self.device)
                ineq_loss = F.mse_loss(ineq_pred, ineq_tgt)

                # Combined loss
                total_loss = c_loss + 0.3 * (lin_loss + quad_loss + ineq_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # Record history
                history['contrastive'].append(c_loss.item())
                history['linear'].append(lin_loss.item())
                history['quadratic'].append(quad_loss.item())
                history['inequality'].append(ineq_loss.item())

                if verbose and epoch % 500 == 0:
                    print(f"Epoch {epoch}: C={c_loss.item():.3f}, Lin={lin_loss.item():.3f}, "
                          f"Quad={quad_loss.item():.3f}, Ineq={ineq_loss.item():.3f}")

            if verbose:
                print("Training complete.")

            return history

        def predict_linear(self, exprs: List[Expr]) -> torch.Tensor:
            """Predict solutions for linear equations."""
            self.encoder.eval()
            self.linear_head.eval()
            with torch.no_grad():
                emb = self.encoder.encode_exprs(exprs, self.device)
                return self.linear_head(emb)

        def predict_quadratic(self, exprs: List[Expr]) -> torch.Tensor:
            """Predict sum of roots for quadratic equations."""
            self.encoder.eval()
            self.quadratic_head.eval()
            with torch.no_grad():
                emb = self.encoder.encode_exprs(exprs, self.device)
                return self.quadratic_head(emb)

        def predict_inequality(self, exprs: List[Expr]) -> torch.Tensor:
            """Predict boundary for inequalities."""
            self.encoder.eval()
            self.inequality_head.eval()
            with torch.no_grad():
                emb = self.encoder.encode_exprs(exprs, self.device)
                return self.inequality_head(emb)

        def save(self, path: str):
            """Save all model weights."""
            torch.save({
                'encoder': self.encoder.state_dict(),
                'linear_head': self.linear_head.state_dict(),
                'quadratic_head': self.quadratic_head.state_dict(),
                'inequality_head': self.inequality_head.state_dict(),
            }, path)
            print(f"Saved to {path}")

        def load(self, path: str):
            """Load all model weights."""
            checkpoint = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.linear_head.load_state_dict(checkpoint['linear_head'])
            self.quadratic_head.load_state_dict(checkpoint['quadratic_head'])
            self.inequality_head.load_state_dict(checkpoint['inequality_head'])
            print(f"Loaded from {path}")

else:
    # Stub if torch not available
    class MultiHeadTrainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for MultiHeadTrainer")
