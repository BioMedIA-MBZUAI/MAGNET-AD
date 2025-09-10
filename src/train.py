"""
MAGNET-AD Training Script
Implements the complete training pipeline following the paper description.

Features:
1. Proper heterogeneous graph construction with 32 brain regions + 100 genes
2. Temporal dynamics through sequential visit processing
3. Multi-task learning with hybrid loss function
4. Survival analysis metrics and PACC regression evaluation
"""

import os
import json
import yaml
import math
import random
import pickle
import argparse
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import HeteroData

# Survival analysis imports
try:
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    print("Warning: lifelines not available, C-index computation will be limited")
    LIFELINES_AVAILABLE = False

from utils import load_csv_data, PatientDataset, collate_hetero_data
from model import MAGNET
from loss import HybridLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_inputs(*args, **kwargs):
    """Validate all inputs before processing"""
    for i, arg in enumerate(args):
        if arg is None:
            raise ValueError(f"Input argument {i} cannot be None")
        if hasattr(arg, 'shape') and 0 in arg.shape:
            raise ValueError(f"Input argument {i} cannot have zero dimensions")
        if hasattr(arg, 'dtype') and 'float' in str(arg.dtype):
            if torch.isnan(arg).any() if hasattr(arg, 'isnan') else np.isnan(arg).any():
                raise ValueError(f"Input argument {i} contains NaN values")
            if torch.isinf(arg).any() if hasattr(arg, 'isinf') else np.isinf(arg).any():
                raise ValueError(f"Input argument {i} contains infinite values")


def load_splits(base_dir: str) -> Dict[str, List[str]]:
    """Load train/val/test splits"""
    with open(os.path.join(base_dir, 'splits.pkl'), 'rb') as f:
        return pickle.load(f)


def load_graphs_for_split(ids: List[str], split_name: str, threshold: int, base_dir: str) -> List[HeteroData]:
    """Load graphs for a specific split"""
    threshold_dir = Path(base_dir) / f'bold_{threshold}'
    graph_file = threshold_dir / f'{split_name}_graphs_bold{threshold}.pkl'
    
    with open(graph_file, 'rb') as f:
        graphs_dict = pickle.load(f)
    
    # Filter out patients with only 1 visit for temporal dynamics
    filtered_graphs = []
    filtered_count = 0
    
    for pid in ids:
        if pid in graphs_dict:
            graph = graphs_dict[pid]
            # Check number of visits
            if hasattr(graph, 'visits') and len(graph.visits) > 1:
                filtered_graphs.append(graph)
            else:
                filtered_count += 1
                
    logger.info(f"Loaded {len(filtered_graphs)} graphs for {split_name}, filtered {filtered_count} single-visit patients")
    return filtered_graphs


def iterate_aligned_batches(graphs: List[HeteroData], dataset: PatientDataset, batch_size: int):
    """Yield aligned (graph_batch, patient_batch) using the same contiguous indices"""
    total = len(graphs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        graph_batch = collate_hetero_data(graphs[start:end])
        patient_batch = dataset[start:end]
        yield graph_batch, patient_batch


def prepare_targets_for_batch(graph_batch: HeteroData) -> torch.Tensor:
    """Prepare PACC targets for a batch of graphs"""
    validate_inputs(graph_batch)
    
    if not hasattr(graph_batch, 'patient_ids'):
        raise ValueError("Graph batch missing 'patient_ids' attribute")
    
    if not hasattr(graph_batch, 'paccv6_scores'):
        raise ValueError("Graph batch missing 'paccv6_scores' attribute")
    
    if len(graph_batch.patient_ids) == 0:
        logger.warning("Empty patient_ids list")
        return torch.tensor([], dtype=torch.float32)
    
    if len(graph_batch.paccv6_scores) == 0:
        logger.warning("Empty paccv6_scores list")
        return torch.tensor([], dtype=torch.float32)
    
    # Each patient has one target (1:1 correspondence)
    if len(graph_batch.patient_ids) != len(graph_batch.paccv6_scores):
        raise ValueError(f"Mismatch: {len(graph_batch.patient_ids)} patients but {len(graph_batch.paccv6_scores)} PACC scores")
    
    # Extract PACC scores directly
    targets = []
    for i, patient_id in enumerate(graph_batch.patient_ids):
        pacc_score = graph_batch.paccv6_scores[i]
        
        # Validate PACC score
        if torch.isnan(pacc_score) or torch.isinf(pacc_score):
            logger.warning(f"Invalid PACC score for patient {patient_id}: {pacc_score}")
            continue
            
        targets.append(pacc_score)
    
    if len(targets) == 0:
        logger.warning("No valid targets found")
        return torch.tensor([], dtype=torch.float32)
    
    return torch.stack(targets)


def compute_cindex(times: np.ndarray, events: np.ndarray, surv_probs: np.ndarray) -> float:
    """Compute concordance index for survival analysis"""
    try:
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available, using simplified C-index computation")
            return 0.5
        
        if len(times) < 2 or np.sum(events) == 0:
            return 0.5
        
        if surv_probs.ndim == 1:
            surv_probs = surv_probs.reshape(-1, 1)
        
        # Use mean survival probability as risk proxy
        risk_scores = np.mean(surv_probs, axis=1)
        
        # Add small noise to break ties
        noise = np.random.normal(0, 1e-8, risk_scores.shape)
        risk_scores = risk_scores + noise
        
        # Validate data
        if (np.any(np.isnan(risk_scores)) or np.any(np.isinf(risk_scores)) or
            np.any(np.isnan(times)) or np.any(np.isinf(times))):
            return 0.5
        
        # Compute C-index (negative risk scores since higher risk = shorter survival)
        c_index = concordance_index(
            event_times=times,
            predicted_scores=-risk_scores,
            event_observed=events
        )
        
        # Validate and clamp result
        if np.isnan(c_index) or np.isinf(c_index):
            return 0.5
        
        return float(np.clip(c_index, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Error computing C-index: {e}")
        return 0.5


def evaluate_model(
    model: MAGNET, 
    graphs: List[HeteroData], 
    dataset: PatientDataset, 
    device: torch.device, 
    batch_size: int
) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    results = {'pacc_mse': [], 'c_index': []}
    
    with torch.no_grad():
        for graph_batch, patient_batch in iterate_aligned_batches(graphs, dataset, batch_size):
            if graph_batch is None:
                continue
                
            graph_batch = graph_batch.to(device)
            if torch.is_tensor(patient_batch):
                patient_batch = patient_batch.to(device)
            
            # Forward pass
            shared_features, deephit_preds, pacc_preds, sequence_preds, temporal_weights = model(
                graph_batch, patient_batch
            )
            
            if deephit_preds is None or pacc_preds is None:
                continue
            
            # Get targets
            targets = prepare_targets_for_batch(graph_batch).to(device)
            
            if len(targets) == 0 or len(pacc_preds) == 0:
                continue
            
            if len(targets) != len(pacc_preds):
                logger.warning(f"Target/prediction mismatch: {len(targets)} vs {len(pacc_preds)}")
                continue
            
            # Calculate MSE
            mse = F.mse_loss(pacc_preds, targets).item()
            results['pacc_mse'].append(mse)
            
            # Calculate C-index using expected time (risk = -E[T])
            try:
                pmf = F.softmax(deephit_preds, dim=1).detach().cpu().numpy()
                bin_indices = np.arange(pmf.shape[1], dtype=np.float32)
                expected_time = (pmf * bin_indices[None, :]).sum(axis=1)
                risk_scores = -expected_time  # higher risk -> lower expected time
                
                times = []
                events = []
                for patient_id in graph_batch.patient_ids:
                    patient_indices = [i for i, pid in enumerate(graph_batch.patient_ids) if pid == patient_id]
                    if patient_indices:
                        last_idx = patient_indices[-1]
                        times.append(graph_batch.survival_times[last_idx].item())
                        events.append(graph_batch.events[last_idx].item())
                
                if len(times) >= 2:
                    times = np.array(times)
                    events = np.array(events)
                    # Align lengths
                    m = min(len(risk_scores), len(times))
                    c_index = compute_cindex(times[:m], events[:m], risk_scores[:m].reshape(-1, 1))
                    results['c_index'].append(c_index)
                
            except Exception as e:
                logger.warning(f"Error computing C-index for batch: {e}")
                continue
    
    if not results['pacc_mse']:
        return {'pacc_mse': float('inf'), 'c_index': 0.0}
    
    return {
        'pacc_mse': np.mean(results['pacc_mse']),
        'c_index': np.mean(results['c_index']) if results['c_index'] else 0.0
    }


def train_epoch(
    model: MAGNET,
    criterion: HybridLoss,
    optimizer: torch.optim.Optimizer,
    graphs: List[HeteroData],
    dataset: PatientDataset,
    device: torch.device,
    batch_size: int,
    grad_clip_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_losses = {
        'total': [],
        'progression': [],
        'pacc': [],
        'temporal': []
    }
    
    num_batches = math.ceil(len(graphs) / batch_size)
    pbar = tqdm(range(num_batches), desc="Training", leave=False)
    
    aligned_iter = iterate_aligned_batches(graphs, dataset, batch_size)
    
    for batch_idx in pbar:
        try:
            graph_batch, patient_batch = next(aligned_iter)
            
            if graph_batch is None:
                continue
                
            graph_batch = graph_batch.to(device)
            if torch.is_tensor(patient_batch):
                patient_batch = patient_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            shared_features, deephit_preds, pacc_preds, sequence_preds, temporal_weights = model(
                graph_batch, patient_batch
            )
            
            if deephit_preds is None or pacc_preds is None:
                continue
            
            # Get targets
            targets = prepare_targets_for_batch(graph_batch).to(device)
            
            if len(targets) == 0:
                logger.warning("No valid targets in batch, skipping")
                continue
            
            # Build time differences for temporal loss
            time_diffs_list = []
            if hasattr(graph_batch, 'patient_visits'):
                for pv in graph_batch.patient_visits:
                    if len(pv) >= 2:
                        # Calculate time differences between consecutive visits
                        diffs = [pv[i+1] - pv[i] for i in range(len(pv)-1)]
                        time_diffs_list.append(torch.tensor(diffs, device=device, dtype=torch.float32))
                    else:
                        time_diffs_list.append(torch.tensor([], device=device))
            else:
                # Fallback: assume unit steps
                for seq_preds in sequence_preds or []:
                    if isinstance(seq_preds, list) and len(seq_preds) > 1:
                        time_diffs_list.append(torch.ones(len(seq_preds)-1, device=device))
                    else:
                        time_diffs_list.append(torch.tensor([], device=device))
            
            # Compute loss
            loss, loss_components = criterion(
                deephit_preds=deephit_preds,
                pacc_preds=pacc_preds,
                survival_times=graph_batch.survival_times,
                events=graph_batch.events,
                pacc_targets=targets,
                sequence_preds=sequence_preds,
                time_diffs=time_diffs_list,
                temporal_weights=temporal_weights
            )
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.warning("NaN gradients detected, skipping batch")
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_components['total_loss'])
            epoch_losses['progression'].append(loss_components['progression_loss'])
            epoch_losses['pacc'].append(loss_components['pacc_loss'])
            epoch_losses['temporal'].append(loss_components['temporal_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_components['total_loss']:.3f}",
                'Prog': f"{loss_components['progression_loss']:.3f}",
                'PACC': f"{loss_components['pacc_loss']:.3f}",
                'Temp': f"{loss_components['temporal_loss']:.3e}",  # Scientific notation for small values
                'α1': f"{loss_components['alpha1']:.2f}",
                'α2': f"{loss_components['alpha2']:.2f}",
                'α3': f"{loss_components['alpha3']:.2f}"
            })
            
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Calculate average losses
    avg_losses = {}
    for key, values in epoch_losses.items():
        if values:
            avg_losses[key] = np.mean(values)
        else:
            avg_losses[key] = 0.0
    
    return avg_losses


def train(config_path: str):
    """Main training function"""
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set up paths
    data_dir = cfg['paths']['data_dir']
    clinical_csv = cfg['paths']['clinical_csv']
    out_dir = Path(cfg['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = cfg['training']['bold_threshold']

    # Load data splits
    splits = load_splits(data_dir)
    train_ids, val_ids, test_ids = splits['train_ids'], splits['val_ids'], splits['test_ids']

    # Load graphs
    train_graphs = load_graphs_for_split(train_ids, 'train_ids', threshold, data_dir)
    val_graphs = load_graphs_for_split(val_ids, 'val_ids', threshold, data_dir)
    test_graphs = load_graphs_for_split(test_ids, 'test_ids', threshold, data_dir)
    
    if not train_graphs:
        raise ValueError("No training graphs loaded")

    # Get patient IDs that were kept after filtering
    train_kept_ids = [graph.patient_id for graph in train_graphs]
    val_kept_ids = [graph.patient_id for graph in val_graphs]
    test_kept_ids = [graph.patient_id for graph in test_graphs]
    
    # Create deterministic ordering
    rng = np.random.default_rng(cfg.get('seed', 42))
    train_order = list(train_kept_ids)
    rng.shuffle(train_order)
    val_order = list(val_kept_ids)
    test_order = list(test_kept_ids)

    # Reorder graphs
    train_graph_map = {g.patient_id: g for g in train_graphs}
    val_graph_map = {g.patient_id: g for g in val_graphs}
    test_graph_map = {g.patient_id: g for g in test_graphs}

    train_graphs = [train_graph_map[pid] for pid in train_order if pid in train_graph_map]
    val_graphs = [val_graph_map[pid] for pid in val_order if pid in val_graph_map]
    test_graphs = [test_graph_map[pid] for pid in test_order if pid in test_graph_map]

    # Load CSV data in same order
    train_csv = load_csv_data(train_order, clinical_csv)
    val_csv = load_csv_data(val_order, clinical_csv)
    test_csv = load_csv_data(test_order, clinical_csv)

    num_features = train_csv.shape[1] if not train_csv.empty else 100

    # Create datasets
    train_dataset = PatientDataset(train_csv, train_order, device=device)
    val_dataset = PatientDataset(val_csv, val_order, device=device)
    test_dataset = PatientDataset(test_csv, test_order, device=device)

    # Infer input dimensions from data if possible to avoid shape mismatches
    inferred_structure_dim = None
    inferred_gene_dim = None
    try:
        probe_graph = train_graphs[0] if len(train_graphs) > 0 else None
        if probe_graph is not None:
            if 'structure' in probe_graph.node_types and hasattr(probe_graph['structure'], 'x'):
                inferred_structure_dim = int(probe_graph['structure'].x.size(1))
            if 'gene' in probe_graph.node_types and hasattr(probe_graph['gene'], 'x'):
                inferred_gene_dim = int(probe_graph['gene'].x.size(1))
    except Exception as e:
        logger.warning(f"Could not infer input dims from graph: {e}")

    structure_input_dim = inferred_structure_dim or cfg['model']['structure_input_dim']
    gene_input_dim = inferred_gene_dim or cfg['model']['gene_input_dim']

    # Initialize model with concrete dims
    model = MAGNET(
        structure_input_dim=structure_input_dim,
        gene_input_dim=gene_input_dim,
        hidden_channels=cfg['model']['hidden_channels'],
        num_spatial_layers=cfg['model']['num_spatial_layers'],
        num_heads=cfg['model']['num_heads'],
        dropout=cfg['model']['dropout'],
        csv_input_dim=num_features,
        csv_hidden_dim=cfg['model']['csv_hidden_dim'],
        csv_output_dim=cfg['model']['csv_output_dim'],
        deephit_duration_index=cfg['model']['deephit_duration_index'],
        use_temporal_dynamics=cfg['model']['use_temporal_dynamics']
    ).to(device)

    logger.info(
        f"Model input dims -> structure: {structure_input_dim}, gene: {gene_input_dim}, csv: {num_features}"
    )

    # Initialize loss function
    criterion = HybridLoss(
        deephit_alpha=cfg['loss']['deephit_alpha'],
        deephit_sigma=cfg['loss']['deephit_sigma'],
        gamma=cfg['loss']['gamma'],
        temporal_gain=cfg['loss']['temporal_gain'],
        temporal_floor=cfg['loss']['temporal_floor']
    ).to(device)

    # Initialize optimizer
    # Coerce optimizer hyperparameters to correct types
    lr = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])
    beta1 = float(cfg['training']['optimizer_beta1'])
    beta2 = float(cfg['training']['optimizer_beta2'])

    # Separate learning rates for alpha weights to prevent collapse
    alpha_lr_scale = float(cfg['training'].get('alpha_lr_scale', 0.1))
    
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters()],
            'lr': lr,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in criterion.named_parameters() if 'log_alpha' in n],
            'lr': lr * alpha_lr_scale,  # Lower LR for alpha weights
            'weight_decay': 0.0  # No weight decay for alpha weights
        },
        {
            'params': [p for n, p in criterion.named_parameters() if 'log_alpha' not in n],
            'lr': lr,
            'weight_decay': weight_decay
        }
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(beta1, beta2)
    )

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # prioritize maximizing C-index
        factor=cfg['scheduler']['factor'],
        patience=cfg['scheduler']['patience'],
        threshold=cfg['scheduler']['threshold'],
        min_lr=cfg['scheduler']['min_lr']
    )

    # Training loop
    best_val_mse = float('inf')
    best_val_cindex = 0.0
    patience_counter = 0
    max_epochs = cfg['training']['max_epochs']
    patience = cfg['training']['patience']

    # Setup logging
    log_path = out_dir / 'train_log.jsonl'
    with open(log_path, 'w') as _:
        pass

    logger.info(f"Starting training for {max_epochs} epochs")
    logger.info(f"Training graphs: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}")

    for epoch in tqdm(range(1, max_epochs + 1), desc="Training", unit="epoch"):
        # Training
        train_losses = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            graphs=train_graphs,
            dataset=train_dataset,
            device=device,
            batch_size=cfg['training']['batch_size'],
            grad_clip_norm=cfg['training']['grad_clip_norm']
        )

        # Validation
        val_metrics = evaluate_model(
            model=model,
            graphs=val_graphs,
            dataset=val_dataset,
            device=device,
            batch_size=cfg['training']['batch_size']
        )

        val_mse = val_metrics['pacc_mse']
        val_c_index = val_metrics['c_index']

        # Learning rate scheduling based on C-index
        scheduler.step(val_c_index)

        # Logging
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Loss: {train_losses['total']:.3f} "
            f"(Prog: {train_losses['progression']:.3f}, "
            f"PACC: {train_losses['pacc']:.3f}, "
            f"Temp: {train_losses['temporal']:.3e}) | "  # Scientific notation for small values
            f"Val MSE: {val_mse:.4f} | "
            f"Val C-index: {val_c_index:.4f}"
        )

        # Track best validation MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
        
        # Early stopping and model saving logic (based on C-index)
        if val_c_index > best_val_cindex:
            best_val_cindex = val_c_index
            patience_counter = 0  # Reset patience counter when we improve
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            logger.info(f"New best model saved - C-index: {val_c_index:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Log to file
        log_entry = {
            'epoch': epoch,
            'train_total_loss': train_losses['total'],
            'train_progression_loss': train_losses['progression'],
            'train_pacc_loss': train_losses['pacc'],
            'train_temporal_loss': train_losses['temporal'],
            'val_pacc_mse': val_mse,
            'val_c_index': val_c_index,
            'lr': optimizer.param_groups[0]['lr'],
            'loss_weights': criterion.get_loss_weights()
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    # Final testing
    logger.info("Testing with best model")
    
    if (out_dir / 'best_model.pth').exists():
        model.load_state_dict(torch.load(out_dir / 'best_model.pth', map_location=device))
    
    test_metrics = evaluate_model(
        model=model,
        graphs=test_graphs,
        dataset=test_dataset,
        device=device,
        batch_size=cfg['training']['batch_size']
    )

    logger.info(f"Test Results:")
    logger.info(f"  Test MSE: {test_metrics['pacc_mse']:.4f}")
    logger.info(f"  Test C-index: {test_metrics['c_index']:.4f}")

    # Save test results
    test_results = {
        'test_mse': test_metrics['pacc_mse'],
        'test_c_index': test_metrics['c_index'],
        'best_val_mse': best_val_mse,
        'best_val_cindex': best_val_cindex,
        'total_epochs': epoch
    }
    
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MAGNET-AD model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    args = parser.parse_args()
    
    train(args.config)
