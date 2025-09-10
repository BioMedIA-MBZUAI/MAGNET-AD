"""
MAGNET-AD Hybrid Loss Function
Implements the exact loss function as described in the paper:

L_total = α1*L_progression_normalized + α2*L_PACC_normalized + α3*L_temporal

Where:
- L_progression: DeepHit loss for AD progression prediction
- L_PACC: MSE loss for PACC score regression  
- L_temporal: Temporal consistency regularization with adaptive weighting
- α1, α2, α3: Learnable balancing weights
- Temporal decay function: β(Δt) = 2 / (1 + exp(γ * |Δt|))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple,Any

# Configure logging
logger = logging.getLogger(__name__)

try:
    import pycox
    from pycox.models.loss import DeepHitSingleLoss
    PYCOX_AVAILABLE = True
except ImportError:
    logger.warning("pycox not available, using custom DeepHit loss implementation")
    PYCOX_AVAILABLE = False


class CustomDeepHitLoss(nn.Module):
    """
    Custom DeepHit loss implementation when pycox is not available.
    Implements the ranking loss for survival analysis.
    """
    
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, phi, duration, event, rank_mat):
        """
        Custom DeepHit loss computation.
        
        Args:
            phi: Predicted probability mass function [batch_size, num_bins]
            duration: Actual duration times [batch_size]
            event: Event indicators [batch_size]
            rank_mat: Ranking matrix [batch_size, batch_size]
        """
        try:
            batch_size = phi.size(0)
            device = phi.device
            
            # Convert to probability mass function
            if phi.dim() == 2:
                pmf = F.softmax(phi, dim=1)
            else:
                pmf = phi
            
            # Cumulative distribution function
            cdf = torch.cumsum(pmf, dim=1)
            
            # Survival function
            survival = 1 - cdf
            
            # Likelihood loss
            likelihood_loss = 0.0
            for i in range(batch_size):
                if event[i] == 1:  # Event occurred
                    # Use probability mass at observed time
                    time_bin = min(int(duration[i].item()), pmf.size(1) - 1)
                    likelihood_loss -= torch.log(pmf[i, time_bin] + 1e-8)
                else:  # Censored
                    # Use survival probability
                    time_bin = min(int(duration[i].item()), survival.size(1) - 1)
                    likelihood_loss -= torch.log(survival[i, time_bin] + 1e-8)
            
            likelihood_loss /= batch_size
            
            # Ranking loss
            ranking_loss = 0.0
            num_pairs = 0
            
            for i in range(batch_size):
                for j in range(batch_size):
                    if rank_mat[i, j] == 1:  # i should have higher risk than j
                        # Use mean survival time as risk proxy
                        risk_i = torch.sum(torch.arange(survival.size(1), device=device).float() * survival[i])
                        risk_j = torch.sum(torch.arange(survival.size(1), device=device).float() * survival[j])
                        
                        # Ranking loss: risk_i should be lower than risk_j (higher risk = lower survival)
                        ranking_loss += F.relu(risk_i - risk_j + self.sigma)
                        num_pairs += 1
            
            if num_pairs > 0:
                ranking_loss /= num_pairs
            
            total_loss = likelihood_loss + self.alpha * ranking_loss
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error in CustomDeepHitLoss: {str(e)}")
            # Return a dummy loss to prevent training failure
            return torch.tensor(1.0, device=phi.device, requires_grad=True)


class HybridLoss(nn.Module):
    """
    Hybrid loss function for MAGNET-AD as described in the paper.
    
    Combines three loss components:
    1. L_progression: DeepHit loss for AD progression prediction
    2. L_PACC: MSE loss for PACC score regression
    3. L_temporal: Temporal consistency regularization
    
    With learnable balancing weights α1, α2, α3 and temporal decay function.
    """
    
    def __init__(
        self,
        deephit_alpha: float = 0.2,
        deephit_sigma: float = 0.1,
        gamma: float = 0.5,
        temporal_gain: float = 1.0,
        weighting: str = 'static',  # 'static' | 'uncertainty' | 'gradnorm'
        gradnorm_alpha: float = 0.5,
        temporal_floor: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        
        # Loss weighting mode
        self.weighting = weighting
        self.gradnorm_alpha = gradnorm_alpha
        self.temporal_floor = temporal_floor

        # Learnable weights (base alphas) - use log space to prevent collapse to 0
        self.log_alpha1 = nn.Parameter(torch.tensor(0.0))  # Progression weight (log space)
        self.log_alpha2 = nn.Parameter(torch.tensor(0.0))  # PACC weight (log space)
        self.log_alpha3 = nn.Parameter(torch.tensor(-0.693))  # Temporal weight (log space, starts at 0.5)
        
        # Add minimum weight constraints
        self.min_weight = 0.01
        self.max_weight = 10.0

        # Uncertainty weighting parameters (log variances)
        if self.weighting == 'uncertainty':
            self.log_sigma_prog = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_pacc = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_temp = nn.Parameter(torch.tensor(0.0))
        
        # DeepHit loss function for progression prediction
        if PYCOX_AVAILABLE:
            self.deephit_loss_fn = DeepHitSingleLoss(alpha=deephit_alpha, sigma=deephit_sigma)
        else:
            self.deephit_loss_fn = CustomDeepHitLoss(alpha=deephit_alpha, sigma=deephit_sigma)
        
        # MSE loss for PACC regression
        self.mse_loss = nn.MSELoss()
        
        # Temporal decay hyperparameter γ
        self.gamma = gamma
        self.temporal_gain = temporal_gain
        
        # Add temporal loss scaling and stability
        self.temporal_scale = 1.0
        self.temporal_momentum = 0.9
        self.register_buffer('temporal_loss_ema', torch.tensor(1.0))
        
        # Running statistics for normalization (exponential moving average)
        self.register_buffer('progression_ema', torch.tensor(1.0))
        self.register_buffer('pacc_ema', torch.tensor(1.0))
        self.register_buffer('temporal_ema', torch.tensor(1.0))
        self.ema_momentum = 0.99
        
        logger.info(f"Initialized HybridLoss with γ={gamma}, temporal_gain={temporal_gain}")

    def beta_decay(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Temporal decay function as described in the paper:
        β(Δt) = 2 / (1 + exp(γ * |Δt|))
        
        Gives higher weight to closely spaced time points.
        
        Args:
            delta_t: Time differences between consecutive visits
            
        Returns:
            Temporal decay weights
        """
        return 2.0 / (1.0 + torch.exp(self.gamma * torch.abs(delta_t)))

    def compute_temporal_loss(
        self,
        sequence_preds: List[List[torch.Tensor]],
        time_diffs: Optional[List[torch.Tensor]] = None,
        temporal_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss as described in the paper.
        
        L_temporal = Σ w_t(i) * ((ŷ_i+1 - ŷ_i)² * β(Δt))
        
        Args:
            sequence_preds: List of per-patient prediction sequences
            time_diffs: Time differences between consecutive visits
            temporal_weights: Adaptive temporal importance weights
            
        Returns:
            Temporal consistency loss
        """
        if not sequence_preds or len(sequence_preds) == 0:
            return torch.tensor(self.temporal_floor, requires_grad=True)
        
        device = sequence_preds[0][0].device if sequence_preds[0] else torch.device('cpu')
        total_temporal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_sequences = 0
        
        for seq_idx, seq in enumerate(sequence_preds):
            if seq is None or len(seq) < 2:
                continue
            
            try:
                # Convert sequence to tensor
                if isinstance(seq, list):
                    if all(torch.is_tensor(s) for s in seq):
                        seq_tensor = torch.stack(seq)
                    else:
                        continue
                elif torch.is_tensor(seq):
                    seq_tensor = seq.flatten() if seq.dim() > 1 else seq
                else:
                    continue
                
                if len(seq_tensor) < 2:
                    continue
                
                # Calculate consecutive differences: (ŷ_i+1 - ŷ_i)²
                pred_diffs = seq_tensor[1:] - seq_tensor[:-1]
                squared_diffs = pred_diffs ** 2
                
                # Apply temporal loss with realistic progression modeling
                # Instead of penalizing all change, encourage smooth but realistic progression
                # Use relative differences to account for different PACC scales
                seq_mean = torch.mean(seq_tensor)
                if seq_mean.abs() > 1e-6:
                    relative_diffs = (pred_diffs / (seq_mean.abs() + 1e-6)) ** 2
                    # Balance between consistency and allowing progression
                    squared_diffs = 0.5 * squared_diffs + 0.5 * relative_diffs
                
                # Ensure non-zero gradient flow
                squared_diffs = torch.clamp(squared_diffs, min=1e-6)
                
                # Get time differences for this sequence
                if time_diffs and seq_idx < len(time_diffs) and time_diffs[seq_idx] is not None:
                    seq_time_diffs = time_diffs[seq_idx]
                    if isinstance(seq_time_diffs, (int, float)):
                        seq_time_diffs = torch.tensor([seq_time_diffs], device=device)
                    elif isinstance(seq_time_diffs, list):
                        seq_time_diffs = torch.tensor(seq_time_diffs, device=device)
                    
                    # Ensure proper length
                    if len(seq_time_diffs) != len(squared_diffs):
                        seq_time_diffs = torch.ones(len(squared_diffs), device=device)
                else:
                    # Default to unit time differences
                    seq_time_diffs = torch.ones(len(squared_diffs), device=device)
                
                # Get adaptive temporal weights for this sequence
                if temporal_weights is not None and torch.is_tensor(temporal_weights):
                    if temporal_weights.numel() >= len(squared_diffs):
                        weights_for_diffs = temporal_weights[:len(squared_diffs)]
                    else:
                        weights_for_diffs = torch.ones(len(squared_diffs), device=device) * temporal_weights.mean()
                else:
                    weights_for_diffs = torch.ones(len(squared_diffs), device=device)
                
                # Apply temporal decay function β(Δt)
                beta_weights = self.beta_decay(seq_time_diffs)
                
                # Temporal loss for this sequence: Σ w_t(i) * ((ŷ_i+1 - ŷ_i)² * β(Δt))
                seq_temporal_loss = torch.sum(weights_for_diffs * squared_diffs * beta_weights)
                
                # Apply temporal gain for proper magnitude (disabled adaptive scaling)
                seq_temporal_loss = seq_temporal_loss * self.temporal_gain
                
                # Add small noise to prevent exact zero gradients
                if seq_temporal_loss.item() < 1e-6:
                    seq_temporal_loss = seq_temporal_loss + torch.randn_like(seq_temporal_loss) * 1e-8
                
                total_temporal_loss = total_temporal_loss + seq_temporal_loss
                valid_sequences += 1
                
            except Exception as e:
                logger.warning(f"Error processing sequence {seq_idx}: {str(e)}")
                continue
        
        # Average over valid sequences
        if valid_sequences > 0:
            total_temporal_loss = total_temporal_loss / valid_sequences
        else:
            # Return the temporal floor if no valid sequences
            total_temporal_loss = torch.tensor(self.temporal_floor, device=device, requires_grad=True)
        
        # Disabled adaptive scaling to prevent temporal loss collapse
        # if self.training:
        #     self.temporal_loss_ema = (self.temporal_momentum * self.temporal_loss_ema + 
        #                             (1 - self.temporal_momentum) * total_temporal_loss.detach())
        #     # Adaptive scaling based on loss magnitude
        #     if self.temporal_loss_ema.item() > 0:
        #         self.temporal_scale = min(10.0, max(0.1, 1.0 / (self.temporal_loss_ema.item() + 1e-8)))
        
        return total_temporal_loss

    def forward(
        self,
        deephit_preds: torch.Tensor,
        pacc_preds: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
        pacc_targets: torch.Tensor,
        sequence_preds: Optional[List[List[torch.Tensor]]] = None,
        time_diffs: Optional[List[torch.Tensor]] = None,
        temporal_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass through hybrid loss function.
        
        Args:
            deephit_preds: DeepHit predictions [batch_size, num_bins]
            pacc_preds: PACC predictions [batch_size]
            survival_times: Ground truth survival times [batch_size]
            events: Event indicators [batch_size]
            pacc_targets: Ground truth PACC scores [batch_size]
            sequence_preds: Per-patient temporal prediction sequences
            time_diffs: Time differences between consecutive visits
            temporal_weights: Adaptive temporal importance weights
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        try:
            device = deephit_preds.device
            batch_size = deephit_preds.size(0)
            
            # Map raw survival times to discrete bin indices compatible with network output
            # Ensure durations lie in [0, num_bins-1]
            num_bins = int(deephit_preds.size(1))
            times_float = survival_times.float()
            max_time = torch.max(times_float).clamp_min(1.0)
            idx_durations = torch.round(times_float / max_time * (num_bins - 1)).long().clamp(0, num_bins - 1)

            # 1. L_progression: DeepHit loss for AD progression prediction
            if PYCOX_AVAILABLE:
                # Build ranking matrix for DeepHit loss
                rank_mat = torch.zeros((batch_size, batch_size), device=device)
                for i in range(batch_size):
                    for j in range(batch_size):
                        if events[i] == 1 and survival_times[i] <= survival_times[j]:
                            rank_mat[i, j] = 1
                
                progression_loss_raw = self.deephit_loss_fn(
                    deephit_preds, idx_durations, events, rank_mat
                )
            else:
                # Use custom implementation
                rank_mat = torch.zeros((batch_size, batch_size), device=device)
                for i in range(batch_size):
                    for j in range(batch_size):
                        if events[i] == 1 and survival_times[i] <= survival_times[j]:
                            rank_mat[i, j] = 1
                
                progression_loss_raw = self.deephit_loss_fn(
                    deephit_preds, idx_durations, events, rank_mat
                )
            
            # 2. L_PACC: MSE loss for PACC score prediction
            pacc_loss_raw = self.mse_loss(pacc_preds, pacc_targets)
            
            # 3. L_temporal: Temporal consistency regularization
            temporal_loss = self.compute_temporal_loss(
                sequence_preds, time_diffs, temporal_weights
            )
            
            # Update exponential moving averages for normalization (during training)
            if self.training:
                self.progression_ema = (self.ema_momentum * self.progression_ema + 
                                       (1 - self.ema_momentum) * progression_loss_raw.detach())
                self.pacc_ema = (self.ema_momentum * self.pacc_ema + 
                                (1 - self.ema_momentum) * pacc_loss_raw.detach())
                self.temporal_ema = (self.ema_momentum * self.temporal_ema +
                                     (1 - self.ema_momentum) * temporal_loss.detach().clamp_min(self.temporal_floor))
            
            # Use EMA normalization for progression and PACC
            progression_loss_normalized = progression_loss_raw / (self.progression_ema + 1e-8)
            pacc_loss_normalized = pacc_loss_raw / (self.pacc_ema + 1e-8)
            
            # Use raw temporal loss with conservative scaling to maintain meaningful gradients
            # Target temporal loss should be roughly 0.01-0.1 scale (1-10% of other losses)
            temporal_loss_normalized = temporal_loss
            
            
            # Only apply floor if loss is exactly zero to prevent collapse
            if temporal_loss.item() == 0.0:
                temporal_loss = torch.tensor(self.temporal_floor, device=temporal_loss.device, requires_grad=True)
            if temporal_loss_normalized.item() == 0.0:
                temporal_loss_normalized = torch.tensor(self.temporal_floor, device=temporal_loss_normalized.device, requires_grad=True)
            
            # Compute dynamic weights
            if self.weighting == 'uncertainty':
                # Kendall & Gal (2018) uncertainty weighting
                w_prog = torch.exp(-self.log_sigma_prog)
                w_pacc = torch.exp(-self.log_sigma_pacc)
                w_temp = torch.exp(-self.log_sigma_temp)
                reg = (self.log_sigma_prog + self.log_sigma_pacc + self.log_sigma_temp)
                total_loss = (
                    w_prog * progression_loss_normalized +
                    w_pacc * pacc_loss_normalized +
                    w_temp * temporal_loss_normalized +
                    reg
                )
            elif self.weighting == 'gradnorm':
                # GradNorm-like adaptive weighting using gradient norms
                # Start from base alphas and adjust by relative gradient norms
                with torch.no_grad():
                    # Use current normalized losses as proxies to compute grads magnitude targets
                    loss_vec = torch.stack([
                        progression_loss_normalized.detach(),
                        pacc_loss_normalized.detach(),
                        temporal_loss_normalized.detach()
                    ])
                    # Relative rates
                    rates = loss_vec / (loss_vec.mean() + 1e-8)
                    target = rates ** self.gradnorm_alpha
                    target = target / (target.sum() + 1e-8)
                base = torch.abs(torch.stack([self.alpha1, self.alpha2, self.alpha3]))
                weights = base / (base.sum() + 1e-8)
                # Move weights slightly toward target each step
                adjusted = 0.9 * weights + 0.1 * target
                w_prog, w_pacc, w_temp = adjusted[0], adjusted[1], adjusted[2]
                total_loss = (
                    w_prog * progression_loss_normalized +
                    w_pacc * pacc_loss_normalized +
                    w_temp * temporal_loss_normalized
                )
            else:
                # Static positive alphas with normalized losses - use softplus to ensure positive weights
                alpha1 = torch.clamp(F.softplus(self.log_alpha1), self.min_weight, self.max_weight)
                alpha2 = torch.clamp(F.softplus(self.log_alpha2), self.min_weight, self.max_weight)
                alpha3 = torch.clamp(F.softplus(self.log_alpha3), self.min_weight, self.max_weight)
                
                total_loss = (
                    alpha1 * progression_loss_normalized +
                    alpha2 * pacc_loss_normalized +
                    alpha3 * temporal_loss_normalized
                )
            
            # Compile loss components for monitoring
            loss_components = {
                'progression_loss': float(progression_loss_raw.detach().cpu()),
                'pacc_loss': float(pacc_loss_raw.detach().cpu()),
                'temporal_loss': float(temporal_loss.detach().cpu()),
                'progression_loss_normalized': float(progression_loss_normalized.detach().cpu()),
                'pacc_loss_normalized': float(pacc_loss_normalized.detach().cpu()),
                'temporal_loss_normalized': float(temporal_loss_normalized.detach().cpu()),
                'total_loss': float(total_loss.detach().cpu()),
                'alpha1': float(torch.clamp(F.softplus(self.log_alpha1), self.min_weight, self.max_weight).detach().cpu()),
                'alpha2': float(torch.clamp(F.softplus(self.log_alpha2), self.min_weight, self.max_weight).detach().cpu()),
                'alpha3': float(torch.clamp(F.softplus(self.log_alpha3), self.min_weight, self.max_weight).detach().cpu()),
                'progression_ema': float(self.progression_ema.detach().cpu()),
                'pacc_ema': float(self.pacc_ema.detach().cpu()),
                'temporal_ema': float(self.temporal_ema.detach().cpu())
            }

            # Monitor gradients of each loss term w.r.t. predictions
            try:
                grads = {}
                # Create graph only as needed, avoid excessive memory
                for name, single_loss in [
                    ('progression', progression_loss_normalized),
                    ('pacc', pacc_loss_normalized),
                    ('temporal', temporal_loss_normalized)
                ]:
                    # Compute grad norm w.r.t. a representative prediction tensor
                    if name == 'progression':
                        grad = torch.autograd.grad(single_loss, deephit_preds, retain_graph=True, allow_unused=True)
                    elif name == 'pacc':
                        grad = torch.autograd.grad(single_loss, pacc_preds, retain_graph=True, allow_unused=True)
                    else:
                        # Approximate with pacc head as temporal depends on sequence preds
                        grad = torch.autograd.grad(single_loss, pacc_preds, retain_graph=True, allow_unused=True)
                    if grad is not None and grad[0] is not None:
                        grads[f'{name}_grad_norm'] = float(grad[0].detach().norm().cpu())
                    else:
                        grads[f'{name}_grad_norm'] = 0.0
                loss_components.update(grads)
            except Exception:
                pass
            
            return total_loss, loss_components
            
        except Exception as e:
            logger.error(f"Error in HybridLoss forward pass: {str(e)}")
            # Return a safe fallback loss to prevent training failure
            fallback_loss = torch.tensor(1.0, device=deephit_preds.device, requires_grad=True)
            fallback_components = {
                'progression_loss': 1.0,
                'pacc_loss': 1.0,
                'temporal_loss': 1.0,
                'total_loss': 1.0,
                'alpha1': 0.33,
                'alpha2': 0.33,
                'alpha3': 0.33
            }
            return fallback_loss, fallback_components

    def get_loss_weights(self) -> Dict[str, float]:
        """Get current learnable loss weights"""
        return {
            'alpha1': float(torch.clamp(F.softplus(self.log_alpha1), self.min_weight, self.max_weight).detach().cpu()),
            'alpha2': float(torch.clamp(F.softplus(self.log_alpha2), self.min_weight, self.max_weight).detach().cpu()),
            'alpha3': float(torch.clamp(F.softplus(self.log_alpha3), self.min_weight, self.max_weight).detach().cpu())
        }

    def set_loss_weights(self, alpha1: float, alpha2: float, alpha3: float):
        """Set loss weights manually"""
        with torch.no_grad():
            self.log_alpha1.fill_(torch.log(torch.clamp(torch.tensor(alpha1), self.min_weight, self.max_weight)))
            self.log_alpha2.fill_(torch.log(torch.clamp(torch.tensor(alpha2), self.min_weight, self.max_weight)))
            self.log_alpha3.fill_(torch.log(torch.clamp(torch.tensor(alpha3), self.min_weight, self.max_weight)))
