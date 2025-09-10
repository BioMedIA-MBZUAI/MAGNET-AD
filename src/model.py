"""
MAGNET-AD: Multi-task Spatiotemporal GNN Model
Following the exact paper description with proper temporal dynamics and loss implementation.

Architecture:
1. Heterogeneous Graph Neural Network with GAT layers
2. Temporal propagation through sequential visits 
3. Multi-task learning: AD progression prediction + PACC regression
4. Hybrid loss function with temporal consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import logging
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional, List, Tuple, Dict

# Configure logging
logger = logging.getLogger(__name__)


class SpatialGraphAttention(nn.Module):
    """
    Spatial Graph Attention Layer for processing heterogeneous graph data.
    Handles structure-structure, gene-gene, and gene-structure relationships.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # Structure-to-structure attention (BOLD correlations)
        self.structure_gat = GATConv(
            in_channels, hidden_channels, 
            heads=num_heads, concat=False, 
            dropout=dropout, edge_dim=1
        )
        
        # Gene-to-gene attention (co-expression)
        self.gene_gat = GATConv(
            in_channels, hidden_channels,
            heads=num_heads, concat=False,
            dropout=dropout, edge_dim=1
        )
        
        # Gene-to-structure cross-modal attention
        self.gene_structure_gat = GATConv(
            in_channels, hidden_channels,
            heads=num_heads, concat=False,
            dropout=dropout, edge_dim=1
        )
        
        # Normalization layers
        self.structure_norm = nn.LayerNorm(hidden_channels)
        self.gene_norm = nn.LayerNorm(hidden_channels)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_structure, x_gene, edge_dict):
        """
        Forward pass through spatial attention layers.
        
        Args:
            x_structure: Structure node features [num_structures, in_channels]
            x_gene: Gene node features [num_genes, in_channels]
            edge_dict: Dictionary of edge indices and attributes
            
        Returns:
            Tuple of processed structure and gene features
        """
        try:
            # 1) Gene-to-gene first
            gene_out = x_gene
            if ('gene', 'coexpressed_with', 'gene') in edge_dict:
                edge_data = edge_dict[('gene', 'coexpressed_with', 'gene')]
                if edge_data['edge_index'].size(1) > 0:
                    gene_out = self.gene_gat(
                        x_gene,
                        edge_data['edge_index'],
                        edge_attr=edge_data['edge_attr']
                    )
                    gene_out = self.gene_norm(gene_out + x_gene)
                    gene_out = F.relu(gene_out)
                    gene_out = self.dropout(gene_out)

            # 2) Gene-to-structure next (cross-modal)
            structure_out = x_structure
            if ('gene', 'expressed_in', 'structure') in edge_dict:
                edge_data = edge_dict[('gene', 'expressed_in', 'structure')]
                if edge_data['edge_index'].size(1) > 0:
                    combined_features = torch.cat([gene_out, structure_out], dim=0)
                    gene_struct_edges = edge_data['edge_index'].clone()
                    gene_struct_edges[1] += gene_out.size(0)
                    cross_modal_out = self.gene_structure_gat(
                        combined_features,
                        gene_struct_edges,
                        edge_attr=edge_data['edge_attr']
                    )
                    gene_size = gene_out.size(0)
                    gene_out = cross_modal_out[:gene_size]
                    structure_out = cross_modal_out[gene_size:]

            # 3) Structure-to-structure last
            if ('structure', 'bold_correlated', 'structure') in edge_dict:
                edge_data = edge_dict[('structure', 'bold_correlated', 'structure')]
                if edge_data['edge_index'].size(1) > 0:
                    updated_structure = self.structure_gat(
                        structure_out,
                        edge_data['edge_index'],
                        edge_attr=edge_data['edge_attr']
                    )
                    structure_out = self.structure_norm(updated_structure + structure_out)
                    structure_out = F.relu(structure_out)
                    structure_out = self.dropout(structure_out)

            return structure_out, gene_out
            
        except Exception as e:
            logger.error(f"Error in SpatialGraphAttention: {str(e)}")
            return x_structure, x_gene


class TemporalPropagation(nn.Module):
    """
    Temporal propagation module for sequential visit processing.
    Implements proper temporal dynamics as described in the paper.
    """
    
    def __init__(self, hidden_channels: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Temporal GAT for edge-aware propagation (prev -> curr)
        # Use learned vector edge attributes derived from 107-d radiomics deltas
        self.temporal_edge_dim = 32
        self.temporal_attr_mlp = nn.Sequential(
            nn.Linear(107, 64),
            nn.ReLU(),
            nn.Linear(64, self.temporal_edge_dim)
        )
        self.temporal_gat = GATConv(
            hidden_channels, hidden_channels, heads=8, concat=False, dropout=dropout, edge_dim=self.temporal_edge_dim
        )
        
        # Temporal fusion layers
        self.temporal_fusion = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal gate for controlling information flow
        self.temporal_gate = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.Sigmoid()
        )

        # Edge-aware temporal weighting MLP: learns a scalar gate from prev, curr, and their difference
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        # Learned edge attribute embedding handled by temporal_attr_mlp

    def forward(self, current_features, previous_features=None, temporal_edges: Optional[Dict[str, torch.Tensor]] = None):
        """
        Forward pass with temporal propagation.
        
        Args:
            current_features: Current visit features [num_nodes, hidden_channels]
            previous_features: Previous visit features [num_nodes, hidden_channels]
            
        Returns:
            Temporally-aware features
        """
        if previous_features is None:
            return current_features
        
        try:
            # If temporal edges are provided, use GAT over bipartite prev->curr graph
            attended_features = None
            if temporal_edges is not None and 'edge_index' in temporal_edges:
                edge_index = temporal_edges['edge_index']  # [2, E] with src in prev, dst in curr (local indices)
                edge_attr = temporal_edges.get('edge_attr', None)  # [E, 107] expected, but may be [E,1] in legacy
                if edge_index is not None and edge_index.numel() > 0:
                    src = edge_index[0].long()
                    dst = edge_index[1].long()
                    prev_src = previous_features[src]  # [E, H]
                    curr_dst = current_features[dst]   # [E, H]
                    delta = curr_dst - prev_src        # [E, H]
                    concat = torch.cat([prev_src, curr_dst, delta], dim=-1)
                    w_change = self.edge_weight_mlp(concat).squeeze(-1)  # [E]
                    if edge_attr is not None:
                        ea = edge_attr.view(edge_attr.size(0), -1)
                        if ea.size(1) == 107:
                            # Learn vector edge features from 107-d delta
                            ea_vec = self.temporal_attr_mlp(ea)  # [E, edge_dim]
                        else:
                            # Fallback for legacy scalar/other sizes: no attribute embedding
                            ea_vec = torch.zeros((ea.size(0), self.temporal_edge_dim), device=current_features.device, dtype=current_features.dtype)
                        # Modulate by change gate
                        edge_feat = ea_vec * w_change.unsqueeze(-1)
                    else:
                        # If missing, fall back to zeros
                        edge_feat = torch.zeros((edge_index.size(1), self.temporal_edge_dim), device=current_features.device, dtype=current_features.dtype)
                    # Use GATConv in bipartite mode with learned vector edge features
                    gat_out = self.temporal_gat((previous_features, current_features), edge_index, edge_attr=edge_feat)
                    attended_features = gat_out

            # Fallback: if no temporal edges, use identity (no update from prev)
            if attended_features is None:
                attended_features = current_features
            
            # Temporal fusion
            combined = torch.cat([current_features, attended_features], dim=-1)
            fused_features = self.temporal_fusion(combined)
            
            # Temporal gate for selective information flow
            gate = self.temporal_gate(combined)
            output = gate * fused_features + (1 - gate) * current_features
            
            return output
            
        except Exception as e:
            logger.error(f"Error in TemporalPropagation: {str(e)}")
            return current_features


class MAGNET(nn.Module):
    """
    MAGNET: Multi-task Spatiotemporal GNN for Alzheimer's Disease Prediction
    
    Implements the complete architecture as described in the paper:
    1. Heterogeneous graph construction with 32 brain regions + 100 genes
    2. Spatial attention layers for intra-timepoint relationships
    3. Temporal propagation for inter-timepoint dynamics
    4. Multi-task prediction heads for progression + PACC
    """
    
    def __init__(
        self,
        structure_input_dim: int = 512,  # 512 if single source; 1024 if [anat512|rad512]
        gene_input_dim: int = 512,       # Projected gene embeddings
        hidden_channels: int = 512,
        num_spatial_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        csv_input_dim: int = 100,
        csv_hidden_dim: int = 64,
        csv_output_dim: int = 32,
        deephit_duration_index: int = 100,
        use_temporal_dynamics: bool = True
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_spatial_layers = num_spatial_layers
        self.use_temporal_dynamics = use_temporal_dynamics
        self.deephit_duration_index = deephit_duration_index
        
        # Input projections with optional fusion gate for [anat|rad]
        self.uses_struct_fusion = (structure_input_dim == 1024)
        if self.uses_struct_fusion:
            self.structure_fusion_gate = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        self.structure_projection = nn.Sequential(
            nn.Linear(512 if self.uses_struct_fusion else structure_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gene_projection = nn.Sequential(
            nn.Linear(gene_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spatial attention layers
        self.spatial_layers = nn.ModuleList([
            SpatialGraphAttention(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_spatial_layers)
        ])
        
        # Temporal propagation module
        if use_temporal_dynamics:
            self.temporal_propagation = TemporalPropagation(
                hidden_channels=hidden_channels,
                dropout=dropout
            )
        
        # Patient clinical data processing
        self.patient_mlp = nn.Sequential(
            nn.Linear(csv_input_dim, csv_hidden_dim),
            nn.LayerNorm(csv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(csv_hidden_dim, csv_output_dim),
            nn.LayerNorm(csv_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature aggregation
        total_feature_dim = hidden_channels * 2 + csv_output_dim  # structure + gene + clinical
        self.feature_aggregator = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        
        # 1. AD Progression Prediction (DeepHit)
        self.progression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, deephit_duration_index)
        )
        
        # 2. PACC Score Regression
        self.pacc_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for regression
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 3. Temporal importance weighting for loss function
        self.temporal_weight_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights properly"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'pacc_head' in name and hasattr(module, 'weight'):
                    # Special initialization for PACC regression head
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def extract_edge_dict(self, data):
        """Extract edge dictionary from HeteroData"""
        edge_dict = {}
        for edge_type in data.edge_types:
            edge_dict[edge_type] = {
                'edge_index': data[edge_type].edge_index,
                'edge_attr': data[edge_type].edge_attr
            }
        return edge_dict

    def process_single_timepoint(self, structure_x, gene_x, edge_dict, previous_structure=None, temporal_edges=None):
        """
        Process a single timepoint through spatial attention layers.
        
        Args:
            structure_x: Structure features [num_structures, hidden_channels]
            gene_x: Gene features [num_genes, hidden_channels]
            edge_dict: Edge information dictionary
            previous_structure: Previous timepoint structure features for temporal propagation
            
        Returns:
            Processed structure and gene features
        """
        try:
            current_structure = structure_x
            current_gene = gene_x
            
            # Apply spatial attention layers
            for layer in self.spatial_layers:
                current_structure, current_gene = layer(
                    current_structure, current_gene, edge_dict
                )
            
            # Apply temporal propagation if enabled and previous features available
            if self.use_temporal_dynamics and previous_structure is not None:
                current_structure = self.temporal_propagation(
                    current_structure, previous_structure, temporal_edges
                )
            
            return current_structure, current_gene
            
        except Exception as e:
            logger.error(f"Error in process_single_timepoint: {str(e)}")
            return structure_x, gene_x

    def forward(self, data, patient_data):
        """
        Forward pass through MAGNET model.
        
        Args:
            data: HeteroData object containing graph information
            patient_data: Patient clinical data tensor
            
        Returns:
            Tuple of (shared_features, progression_preds, pacc_preds, sequence_preds, temporal_weights)
        """
        try:
            if not hasattr(data, 'node_types') or 'structure' not in data.node_types:
                logger.error("Invalid graph data: missing structure nodes")
                return None, None, None, None, None
            
            # Extract edge information (global, will be adapted per visit)
            edge_dict = self.extract_edge_dict(data)
            
            # Get number of patients and visits
            batch_size = len(data.patient_ids) if hasattr(data, 'patient_ids') else 1
            
            # Process each patient separately
            patient_features = []
            sequence_predictions = []
            
            structure_offset = 0
            
            for patient_idx in range(batch_size):
                # Get patient visit information
                if hasattr(data, 'patient_visits') and len(data.patient_visits) > patient_idx:
                    num_visits = len(data.patient_visits[patient_idx])
                else:
                    num_visits = 1
                
                # Determine structures per visit (32 brain regions as per paper)
                num_structures = 32
                
                # Process patient sequentially through visits
                visit_features = []
                previous_structure_features = None
                
                for visit_idx in range(num_visits):
                    # Extract structure features for current visit
                    start_idx = structure_offset + visit_idx * num_structures
                    end_idx = start_idx + num_structures
                    
                    if end_idx <= data['structure'].x.size(0):
                        visit_structure_x = data['structure'].x[start_idx:end_idx]
                        
                        # If using fused [anat|rad]=[512|512], learn α and combine before projection
                        if self.uses_struct_fusion and visit_structure_x.size(1) == 1024:
                            anat = visit_structure_x[:, :512]
                            rad = visit_structure_x[:, 512:]
                            alpha = self.structure_fusion_gate(visit_structure_x)  # [N,1]
                            fused_512 = alpha * anat + (1 - alpha) * rad
                            visit_structure_x = self.structure_projection(fused_512)
                        else:
                            # Direct projection when input is already 512
                            visit_structure_x = self.structure_projection(visit_structure_x)
                        
                        # Get gene features (shared across all visits)
                        if 'gene' in data.node_types and data['gene'].x.size(0) > 0:
                            gene_x = self.gene_projection(data['gene'].x)
                        else:
                            gene_x = torch.zeros((100, self.hidden_channels), device=visit_structure_x.device)
                        
                        # Build visit-local edge dict where structure indices are reindexed to [0, num_structures)
                        visit_edge_dict = {}

                        # 1) Structure-structure (BOLD) edges within this visit only
                        if ('structure', 'bold_correlated', 'structure') in edge_dict:
                            ss = edge_dict[('structure', 'bold_correlated', 'structure')]
                            ss_index = ss['edge_index']
                            ss_attr = ss['edge_attr']
                            # Mask edges where both endpoints lie within [start_idx, end_idx)
                            mask_src = (ss_index[0] >= start_idx) & (ss_index[0] < end_idx)
                            mask_dst = (ss_index[1] >= start_idx) & (ss_index[1] < end_idx)
                            mask = mask_src & mask_dst
                            if mask.any():
                                sub_index = ss_index[:, mask].clone()
                                # Reindex to visit-local [0..num_structures)
                                sub_index[0] -= start_idx
                                sub_index[1] -= start_idx
                                visit_edge_dict[('structure', 'bold_correlated', 'structure')] = {
                                    'edge_index': sub_index,
                                    'edge_attr': ss_attr[mask]
                                }

                        # 2) Gene-gene edges (shared across visits, no reindexing)
                        if ('gene', 'coexpressed_with', 'gene') in edge_dict:
                            gg = edge_dict[('gene', 'coexpressed_with', 'gene')]
                            visit_edge_dict[('gene', 'coexpressed_with', 'gene')] = gg

                        # 3) Gene->structure edges whose structure target lies in this visit
                        if ('gene', 'expressed_in', 'structure') in edge_dict:
                            gs = edge_dict[('gene', 'expressed_in', 'structure')]
                            gs_index = gs['edge_index']
                            gs_attr = gs['edge_attr']
                            # Filter edges with target in [start_idx, end_idx)
                            mask_t = (gs_index[1] >= start_idx) & (gs_index[1] < end_idx)
                            if mask_t.any():
                                sub_index = gs_index[:, mask_t].clone()
                                # Reindex structure target to visit-local
                                sub_index[1] -= start_idx
                                visit_edge_dict[('gene', 'expressed_in', 'structure')] = {
                                    'edge_index': sub_index,
                                    'edge_attr': gs_attr[mask_t]
                                }

                        # Prepare temporal edges from (visit_idx-1) -> visit_idx if available
                        temporal_edges_local = None
                        if visit_idx > 0:
                            # Accept both generic temporally_connected and per-transition temporal_v*_to_v* relations
                            start_prev = structure_offset + (visit_idx - 1) * num_structures
                            end_prev = start_prev + num_structures
                            start_curr = start_idx
                            end_curr = end_idx
                            # Check all edge types for matching temporal relation names
                            for et in edge_dict.keys():
                                if et[0] == 'structure' and et[2] == 'structure' and (
                                    et[1] == 'temporally_connected' or et[1].startswith('temporal_v')
                                ):
                                    tt = edge_dict[et]
                                    tt_index = tt['edge_index']
                                    tt_attr = tt.get('edge_attr', None)
                                    mask_src = (tt_index[0] >= start_prev) & (tt_index[0] < end_prev)
                                    mask_dst = (tt_index[1] >= start_curr) & (tt_index[1] < end_curr)
                                    mask_tt = mask_src & mask_dst
                                    if mask_tt.any():
                                        sub_tt = tt_index[:, mask_tt].clone()
                                        sub_tt[0] -= start_prev  # src relative to prev visit
                                        sub_tt[1] -= start_curr  # dst relative to curr visit
                                        temporal_edges_local = {
                                            'edge_index': sub_tt,
                                            'edge_attr': tt_attr[mask_tt] if (tt_attr is not None) else None
                                        }
                                        break

                        # Process through spatial attention and temporal propagation
                        processed_structure, processed_gene = self.process_single_timepoint(
                            visit_structure_x, gene_x, visit_edge_dict, previous_structure_features, temporal_edges_local
                        )
                        
                        # Store features for temporal propagation
                        previous_structure_features = processed_structure
                        
                        # Pool features for this visit
                        structure_pooled = torch.mean(processed_structure, dim=0)
                        gene_pooled = torch.mean(processed_gene, dim=0)
                        
                        visit_features.append(torch.cat([structure_pooled, gene_pooled], dim=0))
                
                structure_offset += num_visits * num_structures
                
                if not visit_features:
                    continue
                
                # Use features from the last visit for final prediction
                final_features = visit_features[-1]
                
                # Process patient clinical data
                if isinstance(patient_data, torch.Tensor):
                    if patient_data.dim() == 2 and patient_idx < patient_data.size(0):
                        clinical_features = self.patient_mlp(patient_data[patient_idx])
                    elif patient_data.dim() == 1:
                        clinical_features = self.patient_mlp(patient_data)
                    else:
                        clinical_features = torch.zeros(self.patient_mlp[-2].out_features, 
                                                      device=final_features.device)
                else:
                    clinical_features = torch.zeros(self.patient_mlp[-2].out_features, 
                                                  device=final_features.device)
                
                # Combine all features
                combined_features = torch.cat([final_features, clinical_features], dim=0)
                patient_features.append(combined_features)
                
                # Generate sequence predictions for temporal loss
                visit_predictions = []
                for visit_feat in visit_features:
                    temp_combined = torch.cat([visit_feat, clinical_features], dim=0)
                    temp_aggregated = self.feature_aggregator(temp_combined.unsqueeze(0))
                    temp_pacc = self.pacc_head(temp_aggregated).squeeze()
                    visit_predictions.append(temp_pacc)
                
                sequence_predictions.append(visit_predictions)
            
            if not patient_features:
                logger.error("No valid patient features generated")
                return None, None, None, None, None
            
            # Stack patient features
            batch_features = torch.stack(patient_features)
            
            # Feature aggregation
            shared_features = self.feature_aggregator(batch_features)
            
            # Generate predictions
            
            # 1. AD Progression prediction (DeepHit)
            progression_preds = self.progression_head(shared_features)
            
            # 2. PACC score regression
            pacc_preds = self.pacc_head(shared_features).squeeze(-1)
            
            # 3. Temporal importance weights
            temporal_weights = self.temporal_weight_head(shared_features).squeeze(-1)
            
            return shared_features, progression_preds, pacc_preds, sequence_predictions, temporal_weights
            
        except Exception as e:
            logger.error(f"Error in MAGNET forward pass: {str(e)}")
            traceback.print_exc()
            return None, None, None, None, None

    def get_feature_importance(self, data, patient_data):
        """Calculate feature importance scores"""
        try:
            self.eval()
            with torch.no_grad():
                baseline_output = self.forward(data, patient_data)
                if baseline_output[0] is None:
                    return None
                
                baseline_features = baseline_output[0]
                importance_scores = {}
                
                # Structure importance (perturb each brain region)
                structure_importance = []
                for region_idx in range(32):  # 32 brain regions
                    perturbed_data = data.clone()
                    # Zero out features for specific region across all visits
                    region_mask = torch.arange(data['structure'].x.size(0)) % 32 == region_idx
                    perturbed_data['structure'].x[region_mask] = 0
                    
                    perturbed_output = self.forward(perturbed_data, patient_data)
                    if perturbed_output[0] is not None:
                        diff = torch.mean(torch.abs(baseline_features - perturbed_output[0]))
                        structure_importance.append(diff.item())
                    else:
                        structure_importance.append(0.0)
                
                importance_scores['structure'] = structure_importance
                
                # Gene importance (perturb each gene)
                if 'gene' in data.node_types and data['gene'].x.size(0) > 0:
                    gene_importance = []
                    for gene_idx in range(data['gene'].x.size(0)):
                        perturbed_data = data.clone()
                        perturbed_data['gene'].x[gene_idx] = 0
                        
                        perturbed_output = self.forward(perturbed_data, patient_data)
                        if perturbed_output[0] is not None:
                            diff = torch.mean(torch.abs(baseline_features - perturbed_output[0]))
                            gene_importance.append(diff.item())
                        else:
                            gene_importance.append(0.0)
                    
                    importance_scores['gene'] = gene_importance
                
                return importance_scores
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return None
