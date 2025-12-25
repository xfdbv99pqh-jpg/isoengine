# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 2.0.0
# =============================================================================
#
# Spectral regularization for transformer training. Monitors and corrects
# conditioning and rank of weight matrices during training.
#
# Key insight: Frobenius norm penalty does NOT fix conditioning - use
# spectral_norm_sq (penalizes sigma_max^2) which ACTUALLY works.
#
# =============================================================================

import random
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__version__ = "2.0.0"
__all__ = [
    "ASSRConfig",
    "auto_calibrate",
    "compute_stable_rank",
    "compute_stable_rank_ratio",
    "compute_condition_number",
    "compute_spectral_health",
    "compute_spectral_norm_sq_penalty",
    "compute_sv_variance_penalty",
    "print_spectral_report",
    "apply_assr_regularization",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ASSRConfig:
    """
    Configuration for ASSR regularization.

    Args:
        base_lambda: Base regularization strength
        stable_rank_floor: Minimum acceptable stable rank ratio (0-1)
        condition_ceiling: Maximum acceptable condition number
        sample_ratio: Fraction of layers to sample each step (0-1)
        sample_freq: Check every N steps
        max_severity_multiplier: Max scaling factor for adaptive lambda
        subsample_limit: Max matrix dimension for SVD (None = no limit)
        penalty_type: 'spectral_norm_sq' (recommended), 'sv_variance', or 'frobenius'
        log_interventions: Print intervention details
    """

    base_lambda: float = 1e-4
    stable_rank_floor: float = 0.25
    condition_ceiling: float = 500.0
    sample_ratio: float = 0.1
    sample_freq: int = 1
    max_severity_multiplier: float = 10.0
    subsample_limit: Optional[int] = 1024
    condition_severity_scale: float = 10.0
    penalty_type: str = 'spectral_norm_sq'
    log_interventions: bool = False

    def __post_init__(self):
        assert 0 < self.sample_ratio <= 1.0, "sample_ratio must be in (0, 1]"
        assert self.base_lambda >= 0, "base_lambda must be non-negative"
        assert 0 < self.stable_rank_floor < 1, "stable_rank_floor must be in (0, 1)"
        assert self.condition_ceiling > 1, "condition_ceiling must be > 1"
        assert self.penalty_type in ('spectral_norm_sq', 'sv_variance', 'frobenius'), \
            "penalty_type must be 'spectral_norm_sq', 'sv_variance', or 'frobenius'"


if TORCH_AVAILABLE:

    # =========================================================================
    # MATRIX SUBSAMPLING
    # =========================================================================

    def _subsample_matrix(W: torch.Tensor, limit: int) -> torch.Tensor:
        """Subsample large matrix for faster SVD."""
        m, n = W.shape
        if m <= limit and n <= limit:
            return W

        with torch.no_grad():
            if m > limit:
                row_idx = torch.randperm(m, device=W.device)[:limit]
                W = W[row_idx, :]
            if n > limit:
                col_idx = torch.randperm(n, device=W.device)[:limit]
                W = W[:, col_idx]
        return W


    # =========================================================================
    # SPECTRAL METRICS
    # =========================================================================

    def compute_stable_rank(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
        """Compute Stable Rank = ||W||_F^2 / ||W||_2^2"""
        if W.dim() != 2:
            return float(min(W.shape[-2:]) if W.dim() > 2 else W.numel())

        with torch.no_grad():
            try:
                W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
                frob_sq = torch.sum(W_sample ** 2).item()
                spectral_sq = torch.linalg.svdvals(W_sample)[0].item() ** 2
                if spectral_sq > 1e-10:
                    return max(1.0, frob_sq / spectral_sq)
                return 1.0
            except Exception:
                return 1.0


    def compute_stable_rank_ratio(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
        """Compute stable rank as fraction of max possible rank [0, 1]."""
        if W.dim() != 2:
            return 1.0

        if subsample_limit is not None:
            effective_shape = (min(W.shape[0], subsample_limit), min(W.shape[1], subsample_limit))
        else:
            effective_shape = W.shape

        stable_rank = compute_stable_rank(W, subsample_limit)
        max_rank = min(effective_shape[0], effective_shape[1])
        return stable_rank / max_rank if max_rank > 0 else 0.0


    def compute_condition_number(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
        """Compute Condition Number = sigma_max / sigma_min."""
        if W.dim() != 2:
            return 1.0

        with torch.no_grad():
            try:
                W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
                s = torch.linalg.svdvals(W_sample)
                s_max = s[0].item()
                s_min = s[-1].item()
                if s_min > 1e-10:
                    return s_max / s_min
                return float('inf')
            except Exception:
                return float('inf')


    def compute_spectral_health(W: torch.Tensor, subsample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Compute comprehensive spectral health metrics."""
        if W.dim() != 2:
            return {
                'stable_rank': 1.0, 'stable_rank_ratio': 1.0,
                'condition': 1.0, 'spectral_norm': W.norm().item(),
                'shape': tuple(W.shape),
            }

        stable_rank = compute_stable_rank(W, subsample_limit)
        stable_rank_ratio = compute_stable_rank_ratio(W, subsample_limit)
        condition = compute_condition_number(W, subsample_limit)

        with torch.no_grad():
            try:
                W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
                spectral_norm = torch.linalg.svdvals(W_sample)[0].item()
            except Exception:
                spectral_norm = W.norm().item()

        return {
            'stable_rank': stable_rank,
            'stable_rank_ratio': stable_rank_ratio,
            'condition': condition,
            'spectral_norm': spectral_norm,
            'shape': tuple(W.shape),
        }


    # =========================================================================
    # SPECTRAL PENALTIES
    # =========================================================================

    def compute_spectral_norm_sq_penalty(W: torch.Tensor, subsample_limit: Optional[int] = None) -> torch.Tensor:
        """Penalty = sigma_max^2 - ACTUALLY reduces conditioning."""
        if W.dim() != 2:
            return torch.tensor(0.0, device=W.device, dtype=W.dtype)

        W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)

        try:
            s = torch.linalg.svdvals(W_sample)
            return s[0] ** 2
        except Exception:
            return torch.sum(W ** 2)


    def compute_sv_variance_penalty(W: torch.Tensor, subsample_limit: Optional[int] = None) -> torch.Tensor:
        """Penalty = variance of singular values."""
        if W.dim() != 2:
            return torch.tensor(0.0, device=W.device, dtype=W.dtype)

        W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)

        try:
            s = torch.linalg.svdvals(W_sample)
            return ((s - s.mean()) ** 2).mean()
        except Exception:
            return torch.tensor(0.0, device=W.device, dtype=W.dtype)


    def _compute_penalty(W: torch.Tensor, penalty_type: str, subsample_limit: Optional[int] = None) -> torch.Tensor:
        """Compute penalty based on type."""
        if penalty_type == 'spectral_norm_sq':
            return compute_spectral_norm_sq_penalty(W, subsample_limit)
        elif penalty_type == 'sv_variance':
            return compute_sv_variance_penalty(W, subsample_limit)
        else:
            return torch.sum(W ** 2)


    # =========================================================================
    # AUTO-CALIBRATION
    # =========================================================================

    def auto_calibrate(
        model: nn.Module,
        percentile: float = 10.0,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> ASSRConfig:
        """Auto-calibrate ASSR parameters based on model size and spectral stats."""
        if seed is not None:
            random.seed(seed)

        linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        num_layers = len(linear_layers)

        if num_layers == 0:
            if verbose:
                print("[ASSR] No linear layers found, using defaults")
            return ASSRConfig()

        total_params = sum(p.numel() for p in model.parameters())

        # Model size tiers
        if total_params < 100_000_000:
            sample_ratio, sample_freq, subsample_limit, base_lambda = 0.10, 1, 1024, 1e-4
            tier = "Small (<100M)"
        elif total_params < 500_000_000:
            sample_ratio, sample_freq, subsample_limit, base_lambda = 0.05, 2, 768, 5e-5
            tier = "Medium (100M-500M)"
        else:
            sample_ratio, sample_freq, subsample_limit, base_lambda = 0.02, 5, 512, 2e-5
            tier = "Large (>500M)"

        # Collect spectral stats
        sr_values, cond_values = [], []
        calibration_layers = random.sample(linear_layers, min(50, num_layers)) if num_layers > 50 else linear_layers

        for name, m in calibration_layers:
            sr_values.append(compute_stable_rank_ratio(m.weight, subsample_limit))
            cond = compute_condition_number(m.weight, subsample_limit)
            if cond < float('inf'):
                cond_values.append(cond)

        sr_arr = np.array(sr_values)
        cond_arr = np.array(cond_values) if cond_values else np.array([500.0])

        sr_floor = max(0.05, min(float(np.percentile(sr_arr, percentile) * 0.8), 0.4))
        cond_ceiling = max(100, float(np.percentile(cond_arr, 100 - percentile) * 1.5))

        if verbose:
            print(f"\n[ASSR] Auto-Calibration v{__version__}")
            print(f"   Model tier: {tier}")
            print(f"   Parameters: {total_params/1e6:.1f}M | Linear layers: {num_layers}")
            print(f"   sample_ratio={sample_ratio} | condition_ceiling={cond_ceiling:.0f}")
            print(f"   SR ratio: [{sr_arr.min():.3f}, {np.median(sr_arr):.3f}, {sr_arr.max():.3f}]")
            print(f"   Condition: [{cond_arr.min():.0f}, {np.median(cond_arr):.0f}, {cond_arr.max():.0f}]\n")

        return ASSRConfig(
            base_lambda=base_lambda,
            stable_rank_floor=sr_floor,
            condition_ceiling=cond_ceiling,
            sample_ratio=sample_ratio,
            sample_freq=sample_freq,
            subsample_limit=subsample_limit,
        )


    # =========================================================================
    # REGULARIZATION FUNCTION (for custom training loops)
    # =========================================================================

    def apply_assr_regularization(
        model: nn.Module,
        config: ASSRConfig,
        linear_layers: Optional[List[nn.Module]] = None,
    ) -> torch.Tensor:
        """
        Apply ASSR regularization to a model.

        Use this in custom training loops:
            loss = criterion(output, target)
            assr_loss = apply_assr_regularization(model, assr_config)
            total_loss = loss + assr_loss
            total_loss.backward()

        Args:
            model: The model being trained
            config: ASSR configuration
            linear_layers: Optional cached list of linear layers

        Returns:
            Regularization loss tensor (add to main loss)
        """
        if linear_layers is None:
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        if not linear_layers:
            return torch.tensor(0.0)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        reg_loss = torch.tensor(0.0, device=device, dtype=dtype)

        num_sample = max(1, int(len(linear_layers) * config.sample_ratio))
        subset = random.sample(linear_layers, min(num_sample, len(linear_layers)))

        for m in subset:
            W = m.weight
            limit = config.subsample_limit

            # Check condition number
            cond = compute_condition_number(W, limit)
            if cond > config.condition_ceiling:
                severity = min(max((cond / config.condition_ceiling - 1) / config.condition_severity_scale, 0), 1)
                adaptive_lambda = config.base_lambda * (1 + config.max_severity_multiplier * severity)
                penalty = _compute_penalty(W, config.penalty_type, limit)
                reg_loss = reg_loss + adaptive_lambda * penalty

            # Check stable rank
            sr = compute_stable_rank_ratio(W, limit)
            if sr < config.stable_rank_floor:
                severity = min(max((config.stable_rank_floor - sr) / config.stable_rank_floor, 0), 1)
                adaptive_lambda = config.base_lambda * (1 + config.max_severity_multiplier * severity)
                penalty = compute_sv_variance_penalty(W, limit)
                reg_loss = reg_loss + adaptive_lambda * penalty

        return reg_loss


    # =========================================================================
    # UTILITIES
    # =========================================================================

    def print_spectral_report(model: nn.Module, top_k: int = 10, subsample_limit: Optional[int] = 1024) -> None:
        """Print spectral health report for all linear layers."""
        results = []

        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                health = compute_spectral_health(m.weight, subsample_limit)
                results.append({'name': name, **health})

        print("\n" + "=" * 70)
        print("  SPECTRAL HEALTH REPORT")
        print("=" * 70)
        print(f"  Linear Layers: {len(results)}")

        if not results:
            print("  No linear layers found.")
            return

        sr_vals = [r['stable_rank_ratio'] for r in results]
        c_vals = [r['condition'] for r in results if r['condition'] < float('inf')]

        print(f"  SR Ratio: [{min(sr_vals):.3f}, {np.median(sr_vals):.3f}, {max(sr_vals):.3f}]")
        if c_vals:
            print(f"  Condition: [{min(c_vals):.0f}, {np.median(c_vals):.0f}, {max(c_vals):.0f}]")

        results.sort(key=lambda x: x['stable_rank_ratio'])

        print(f"\n  {'Layer':<40} {'Shape':<15} {'SR':>8} {'Cond':>10}")
        print("  " + "-" * 73)

        for r in results[:top_k]:
            name = r['name'][-38:] if len(r['name']) > 38 else r['name']
            cond_str = f"{r['condition']:.0f}" if r['condition'] < 1e6 else "inf"
            print(f"  {name:<40} {str(r['shape']):<15} {r['stable_rank_ratio']:>8.3f} {cond_str:>10}")

        print("=" * 70 + "\n")

else:
    # Stubs if torch not available
    def compute_stable_rank(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def compute_stable_rank_ratio(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def compute_condition_number(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def compute_spectral_health(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def compute_spectral_norm_sq_penalty(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def compute_sv_variance_penalty(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def auto_calibrate(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def apply_assr_regularization(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")

    def print_spectral_report(*args, **kwargs):
        raise RuntimeError("PyTorch required for ASSR")
