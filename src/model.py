import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
from torch_geometric.nn import GCNConv
import copy


class STBlock(nn.Module):
    """Spatio-Temporal Block for processing graph data with attention mechanisms."""
    
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.1, temporal_dim=64,
                 use_gene_edges=True, use_gene_nodes=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.use_gene_edges = use_gene_edges
        self.use_gene_nodes = use_gene_nodes

        # Structure connection layers
        self.bold_gcn = GCNConv(-1, hidden_channels)
        self.temporal_gcn = GCNConv(-1, hidden_channels)

        # Gene-related layers only if using gene nodes
        if self.use_gene_nodes:
            self.gene_gene_gcn = GCNConv(-1, hidden_channels)
            self.gene_struct_gcn = GCNConv(-1, hidden_channels)

            # Cross attention layers
            self.structure_gene_attention = nn.MultiheadAttention(
                hidden_channels, num_heads, dropout=dropout, batch_first=True
            )
            self.gene_structure_attention = nn.MultiheadAttention(
                in_channels, num_heads, dropout=dropout, batch_first=True
            )

        # Temporal edge projection
        self.temporal_edge_proj = nn.Sequential(
            nn.Linear(107, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, temporal_dim)
        )

        # Normalizations
        self.struct_norm = nn.LayerNorm(hidden_channels)
        self.temporal_norm = nn.LayerNorm(hidden_channels)
        
        if self.use_gene_nodes:
            self.gene_norm = nn.LayerNorm(hidden_channels)
            self.stage_norms = nn.ModuleDict({
                'early': nn.LayerNorm(hidden_channels),
                'mid': nn.LayerNorm(hidden_channels),
                'late': nn.LayerNorm(hidden_channels)
            })

            self.final_projection = nn.Linear(in_channels + hidden_channels, hidden_channels)
            self.final_norm = nn.LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_struct, x_gene, edge_dict, time_steps=1):
        """
        Forward pass through the STBlock.
        
        Args:
            x_struct: Structure node features
            x_gene: Gene node features
            edge_dict: Dictionary of edges by type
            time_steps: Number of time steps in the sequence
            
        Returns:
            tuple: Processed structure features, gene features, and combined features
        """
        try:
            # Process structure features first
            struct_features = self._process_spatial_temporal(x_struct, edge_dict)

            if self.use_gene_nodes:
                # Process gene-gene and gene-structure connections
                gene_features = self._process_gene_gene(x_gene, edge_dict)
                struct_features = self._process_gene_structure(struct_features, gene_features, edge_dict)
                
                # Process cross-modal attention
                struct_attended, gene_attended, combined = self._process_cross_modal_attention(
                    struct_features, gene_features, edge_dict
                )
                return struct_attended, gene_attended, combined
            else:
                # Return only structure features if not using gene nodes
                return struct_features, None, struct_features

        except Exception as e:
            print(f"Error in STBlock forward pass: {str(e)}")
            traceback.print_exc()
            return None, None, None

    def _process_gene_gene(self, x_gene, edge_dict):
        """Process gene-gene interactions if enabled."""
        if not self.use_gene_nodes or not self.use_gene_edges:
            return x_gene

        gene_features = x_gene.float()
        original_gene = gene_features

        for stage in ['early', 'mid', 'late']:
            gene_key = ('gene', f'interacts_{stage}', 'gene')
            if gene_key in edge_dict and edge_dict[gene_key]['edge_index'].size(1) > 0:
                edge_index = edge_dict[gene_key]['edge_index']
                edge_weight = edge_dict[gene_key]['edge_attr'].float().squeeze(-1)

                stage_out = self.gene_gene_gcn(gene_features, edge_index, edge_weight)
                gene_features = stage_out + gene_features
                gene_features = self.stage_norms[stage](gene_features)
                gene_features = F.relu(gene_features)

        return self.gene_norm(gene_features + original_gene)

    def _process_spatial_temporal(self, x_struct, edge_dict):
        """Process spatial (BOLD) and temporal connections."""
        struct_features = x_struct.float()
        original_struct = struct_features

        # Process BOLD correlations
        bold_key = ('structure', 'bold_correlated', 'structure')
        if bold_key in edge_dict and edge_dict[bold_key]['edge_index'].size(1) > 0:
            edge_index = edge_dict[bold_key]['edge_index']
            edge_weight = edge_dict[bold_key]['edge_attr'].float().squeeze(-1)

            bold_out = self.bold_gcn(struct_features, edge_index, edge_weight)
            struct_features = bold_out + struct_features
            struct_features = self.struct_norm(struct_features)
            struct_features = F.relu(struct_features)

        # Process temporal connections
        temporal_key = ('structure', 'temporally_connected', 'structure')
        if temporal_key in edge_dict and edge_dict[temporal_key]['edge_index'].size(1) > 0:
            temp_edge_index = edge_dict[temporal_key]['edge_index']
            temp_edge_attr = edge_dict[temporal_key]['edge_attr'].float()

            # Handle edge feature dimensionality
            if temp_edge_attr.dim() == 1:
                temp_edge_attr = temp_edge_attr.unsqueeze(-1)
                
            # If edge features don't match expected size, use a default weight
            if temp_edge_attr.size(1) != 107:
                temp_weight = torch.ones_like(temp_edge_index[0], dtype=torch.float32)
            else:
                temp_weight = self.temporal_edge_proj(temp_edge_attr).squeeze(-1)
                
            temporal_out = self.temporal_gcn(struct_features, temp_edge_index, temp_weight)
            struct_features = temporal_out + struct_features
            struct_features = self.temporal_norm(struct_features)
            struct_features = F.relu(struct_features)

        return self.struct_norm(struct_features + original_struct)

    def _process_gene_structure(self, x_struct, gene_features, edge_dict):
        """Process gene-structure connections with proper index bounds."""
        if not self.use_gene_nodes:
            return x_struct

        struct_features = x_struct.float()
        original_struct = struct_features

        if len(struct_features.shape) == 3:
            batch_size, num_structs, hidden_dim = struct_features.shape
        else:
            num_structs = struct_features.size(0)
            struct_features = struct_features.unsqueeze(0)
            batch_size = 1

        if len(gene_features.shape) == 3:
            _, num_genes, _ = gene_features.shape
        else:
            num_genes = gene_features.size(0)
            gene_features = gene_features.unsqueeze(0)

        for stage in ['early', 'mid', 'late']:
            key = ('gene', f'connects_{stage}', 'structure')
            if key in edge_dict and edge_dict[key]['edge_index'].size(1) > 0:
                edge_index = edge_dict[key]['edge_index']
                edge_attr = edge_dict[key]['edge_attr']

                valid_indices = []
                valid_weights = []

                for i in range(edge_index.size(1)):
                    gene_idx = edge_index[0, i].item()
                    struct_idx = edge_index[1, i].item()
                    
                    # Extract weight safely
                    if edge_attr.dim() > 1:
                        weight = edge_attr[i, 0].item()
                    else:
                        weight = edge_attr[i].item()

                    if gene_idx < num_genes and struct_idx < num_structs:
                        valid_indices.append([gene_idx, struct_idx])
                        valid_weights.append(weight)

                if valid_indices:
                    valid_edge_index = torch.tensor(valid_indices, device=edge_index.device).t()
                    valid_edge_weights = torch.tensor(valid_weights, device=edge_attr.device)

                    try:
                        x_combined = gene_features.squeeze(0)
                        gcn_out = self.gene_struct_gcn(
                            x_combined,
                            valid_edge_index,
                            valid_edge_weights
                        )

                        struct_out = struct_features.clone()
                        for i in range(valid_edge_index.size(1)):
                            gene_idx = valid_edge_index[0, i]
                            struct_idx = valid_edge_index[1, i]
                            weight = valid_edge_weights[i]
                            struct_out[0, struct_idx] += weight * gcn_out[gene_idx]

                        struct_features = self.stage_norms[stage](struct_out)
                        struct_features = F.relu(struct_features)

                    except Exception as e:
                        print(f"Error in {stage} stage GCN: {str(e)}")
                        continue

        return self.struct_norm(struct_features + original_struct)

    def _process_cross_modal_attention(self, structure_feat, gene_feat, edge_dict):
        """Process cross-modal attention if gene nodes are enabled."""
        if not self.use_gene_nodes:
            return structure_feat, None, structure_feat

        try:
            if len(structure_feat.shape) == 2:
                structure_feat = structure_feat.unsqueeze(0)
            if len(gene_feat.shape) == 2:
                gene_feat = gene_feat.unsqueeze(0)

            batch_size = structure_feat.size(0)
            num_structures = structure_feat.size(1)
            num_genes = gene_feat.size(1)

            struct_to_gene_base = torch.zeros(
                num_structures, num_genes,
                device=structure_feat.device
            )
            gene_to_struct_base = torch.zeros(
                num_genes, num_structures,
                device=gene_feat.device
            )

            has_valid_edges = False

            for stage in ['early', 'mid', 'late']:
                key = ('gene', f'connects_{stage}', 'structure')
                if key in edge_dict and edge_dict[key]['edge_index'].size(1) > 0:
                    edge_index = edge_dict[key]['edge_index']
                    edge_weight = edge_dict[key]['edge_attr'].float().squeeze(-1)

                    valid_genes = edge_index[0] < num_genes
                    valid_structs = edge_index[1] < num_structures
                    valid_mask = valid_genes & valid_structs

                    if valid_mask.any():
                        has_valid_edges = True
                        edge_index = edge_index[:, valid_mask]
                        edge_weight = edge_weight[valid_mask]

                        for i in range(edge_index.size(1)):
                            gene_idx = edge_index[0, i].item()
                            struct_idx = edge_index[1, i].item()
                            weight = edge_weight[i].item()
                            struct_to_gene_base[struct_idx, gene_idx] = weight
                            gene_to_struct_base[gene_idx, struct_idx] = weight

            if not has_valid_edges:
                struct_attended, _ = self.structure_gene_attention(
                    structure_feat,
                    gene_feat,
                    gene_feat
                )
                gene_attended, _ = self.gene_structure_attention(
                    gene_feat,
                    structure_feat,
                    structure_feat
                )
            else:
                struct_to_gene_base = torch.where(
                    struct_to_gene_base > 0,
                    struct_to_gene_base,
                    torch.tensor(-1e9, device=struct_to_gene_base.device)
                )
                gene_to_struct_base = torch.where(
                    gene_to_struct_base > 0,
                    gene_to_struct_base,
                    torch.tensor(-1e9, device=gene_to_struct_base.device)
                )

                struct_to_gene_mask = struct_to_gene_base.unsqueeze(0).expand(
                    self.num_heads, -1, -1
                )
                gene_to_struct_mask = gene_to_struct_base.unsqueeze(0).expand(
                    self.num_heads, -1, -1
                )

                struct_attended, _ = self.structure_gene_attention(
                    structure_feat,
                    gene_feat,
                    gene_feat,
                    attn_mask=struct_to_gene_mask
                )

                gene_attended, _ = self.gene_structure_attention(
                    gene_feat,
                    structure_feat,
                    structure_feat,
                    attn_mask=gene_to_struct_mask
                )

            struct_pooled = struct_attended.mean(1)
            gene_pooled = gene_attended.mean(1)

            combined = torch.cat([struct_pooled, gene_pooled], dim=-1)
            combined = self.final_projection(combined)
            combined = self.final_norm(combined)

            return struct_attended, gene_attended, combined

        except Exception as e:
            print(f"Error in cross modal attention: {str(e)}")
            traceback.print_exc()
            return None, None, None


class BrainTemporalGNN(nn.Module):
    """
    Graph Neural Network for brain temporal data analysis with DeepHit survival prediction.
    """
    
    def __init__(self, in_channels=512, hidden_channels=512, num_blocks=3, 
                csv_input_dim=100, csv_hidden_dim=64, csv_output_dim=32,
                dropout=0.2, deephit_duration_index=2305, st_num_heads=8, st_dropout=0.2,
                use_gene_edges=True, use_gene_nodes=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_gene_edges = use_gene_edges
        self.use_gene_nodes = use_gene_nodes
        self.deephit_duration_index = deephit_duration_index

        # Define structures for brain regions
        self.structures = [
            "left_cerebral_white_matter", "left_cerebral_cortex", "left_lateral_ventricle",
            "left_inferior_lateral_ventricle", "left_cerebellum_white_matter", "left_cerebellum_cortex",
            "left_thalamus", "left_caudate", "left_putamen", "left_pallidum", "third_ventricle",
            "fourth_ventricle", "brain_stem", "left_hippocampus", "left_amygdala",
            "left_accumbens_area", "csf", "left_ventral_dc", "right_cerebral_white_matter",
            "right_cerebral_cortex", "right_lateral_ventricle", "right_inferior_lateral_ventricle",
            "right_cerebellum_white_matter", "right_cerebellum_cortex", "right_thalamus",
            "right_caudate", "right_putamen", "right_pallidum", "right_hippocampus",
            "right_amygdala", "right_accumbens_area", "right_ventral_dc"
        ]

        # Initial feature projections with LayerNorm
        self.structure_proj = nn.Sequential(
            nn.Linear(512, hidden_channels), 
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.use_gene_nodes:
            self.gene_proj = nn.Sequential(
                nn.Linear(512, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Patient MLP
        self.patient_mlp = nn.Sequential(
            nn.Linear(csv_input_dim, csv_hidden_dim),
            nn.LayerNorm(csv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(csv_hidden_dim, csv_hidden_dim // 2),
            nn.LayerNorm(csv_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(csv_hidden_dim // 2, csv_output_dim)
        )

        # ST blocks
        self.st_blocks = nn.ModuleList([
            STBlock(in_channels, hidden_channels, dropout=st_dropout, num_heads=st_num_heads,
                   use_gene_edges=use_gene_edges, use_gene_nodes=use_gene_nodes)
            for _ in range(num_blocks)
        ])

        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            hidden_channels, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )

        # Feature aggregation input dimension based on gene node usage
        total_dim = hidden_channels + csv_output_dim
        if self.use_gene_nodes:
            total_dim += 2 * hidden_channels  # Add gene feature dimensions

        self.feature_aggregator = nn.Sequential(
            nn.Linear(total_dim, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # DeepHit network
        self.deephit_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LayerNorm(hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, deephit_duration_index)
        )

        self.paccv6_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LayerNorm(hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1)
        )

    def _prepare_edge_dict(self, data, start_idx, end_idx, current_stages, visit_idx, total_visits):
        """Prepare edge dictionary with gene control flags."""
        edge_dict = {}
        structures_per_visit = len(self.structures)

        try:
            # Structure-structure BOLD correlations
            bold_key = ('structure', 'bold_correlated', 'structure')
            if bold_key in data.edge_types:
                edge_index = data[bold_key].edge_index
                edge_attr = data[bold_key].edge_attr

                mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & 
                    (edge_index[1] >= start_idx) & (edge_index[1] < end_idx))

                valid_edge_index = edge_index[:, mask]
                valid_edge_attr = edge_attr[mask]
                valid_edge_index = valid_edge_index - start_idx

                edge_dict[bold_key] = {
                    'edge_index': valid_edge_index,
                    'edge_attr': valid_edge_attr
                }

            # Temporal connections
            if visit_idx > 0:
                temp_key = ('structure', 'temporally_connected', 'structure')
                if temp_key in data.edge_types:
                    edge_index = data[temp_key].edge_index
                    edge_attr = data[temp_key].edge_attr

                    prev_start = start_idx - structures_per_visit
                    mask = ((edge_index[0] >= start_idx) & (edge_index[0] < end_idx) & 
                        (edge_index[1] >= prev_start) & (edge_index[1] < start_idx))

                    valid_edge_index = edge_index[:, mask]
                    valid_edge_attr = edge_attr[mask]

                    valid_edge_index = valid_edge_index.clone()
                    valid_edge_index[0] = valid_edge_index[0] - start_idx
                    valid_edge_index[1] = valid_edge_index[1] - prev_start

                    edge_dict[temp_key] = {
                        'edge_index': valid_edge_index,
                        'edge_attr': valid_edge_attr
                    }

            # Gene-related edges only if enabled
            if self.use_gene_nodes:
                for stage in current_stages:
                    # Gene-gene interactions if both flags are True
                    if self.use_gene_edges:
                        gene_key = ('gene', f'interacts_{stage}', 'gene')
                        if gene_key in data.edge_types:
                            edge_dict[gene_key] = {
                                'edge_index': data[gene_key].edge_index,
                                'edge_attr': data[gene_key].edge_attr
                            }

                    # Gene-structure connections
                    gene_struct_key = ('gene', f'connects_{stage}', 'structure')
                    if gene_struct_key in data.edge_types:
                        edge_index = data[gene_struct_key].edge_index.clone()
                        edge_attr = data[gene_struct_key].edge_attr

                        mask = (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
                        valid_edge_index = edge_index[:, mask]
                        valid_edge_attr = edge_attr[mask]

                        valid_edge_index[1] = valid_edge_index[1] - start_idx

                        edge_dict[gene_struct_key] = {
                            'edge_index': valid_edge_index,
                            'edge_attr': valid_edge_attr
                        }

            return edge_dict

        except Exception as e:
            print(f"Error preparing edge dictionary: {str(e)}")
            traceback.print_exc()
            return {}

    def _process_patient_features(self, data, patient_data, patient_id, patient_indices, visit_indices):
        """Process features for a single patient with gene control flags."""
        try:
            structures_per_visit = len(self.structures)
            processed_structs = []
            num_visits = len(patient_indices)

            for visit_idx, start_idx in enumerate(patient_indices):
                end_idx = start_idx + structures_per_visit

                # Process structure features
                patient_struct = data['structure'].x[start_idx:end_idx]
                processed_time_struct = self.structure_proj(patient_struct)
                processed_time_struct = F.relu(processed_time_struct)
                processed_structs.append(processed_time_struct)

                # Determine stages based on number of visits
                if num_visits == 1:
                    current_stages = ["early", "mid", "late"]
                elif num_visits == 2:
                    current_stages = ["early", "mid"] if visit_idx == 0 else ["late"]
                else:
                    if visit_idx == 0:
                        current_stages = ["early"]
                    elif visit_idx == num_visits - 1:
                        current_stages = ["late"]
                    elif visit_idx == num_visits // 2:
                        current_stages = ["mid"]
                    else:
                        current_stages = []

                edge_dict = self._prepare_edge_dict(
                    data=data,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    current_stages=current_stages,
                    visit_idx=visit_idx,
                    total_visits=num_visits
                )

                all_struct_features = []
                all_gene_features = []
                all_combined_features = []

                for block in self.st_blocks:
                    if self.use_gene_nodes and 'gene' in data.node_types:
                        struct_attended, gene_attended, combined = block(
                            processed_time_struct.unsqueeze(0),
                            data['gene'].x.unsqueeze(0),
                            edge_dict,
                            time_steps=num_visits
                        )
                    else:
                        # Skip gene processing if not using gene nodes
                        struct_attended = block(
                            processed_time_struct.unsqueeze(0),
                            None,
                            edge_dict,
                            time_steps=num_visits
                        )[0]
                        gene_attended = None
                        combined = struct_attended

                    if struct_attended is not None:
                        if not torch.isnan(struct_attended).any():
                            all_struct_features.append(struct_attended)
                            if self.use_gene_nodes and gene_attended is not None:
                                all_gene_features.append(gene_attended)
                                all_combined_features.append(combined)

            if not all_struct_features:
                return None

            # Process structure features
            final_struct = torch.stack(all_struct_features).mean(0)
            struct_feat = final_struct.mean(dim=0).mean(dim=0).unsqueeze(0)

            # Process gene features if enabled
            if self.use_gene_nodes and all_gene_features and combined is not None:
                final_gene = torch.stack(all_gene_features).mean(0)
                final_combined = torch.stack(all_combined_features).mean(0)
                gene_feat = final_gene.mean(dim=0).mean(dim=0).unsqueeze(0)
                combined_feat = final_combined.mean(dim=0).unsqueeze(0)
            else:
                gene_feat = None
                combined_feat = struct_feat

            # Process patient data
            try:
                # Get patient index in the dataset
                if isinstance(patient_id, str) and patient_id in data.patient_ids:
                    patient_idx = list(data.patient_ids).index(patient_id)
                elif hasattr(patient_data, 'patient_ids') and patient_id in patient_data.patient_ids:
                    patient_idx = list(patient_data.patient_ids).index(patient_id)
                else:
                    patient_idx = 0  # Default if not found

                # Get patient features
                if isinstance(patient_data, torch.Tensor):
                    if patient_data.dim() == 2:
                        patient_out = self.patient_mlp(patient_data[patient_idx])
                    else:
                        patient_out = self.patient_mlp(patient_data)
                else:  # Handle list or other sequence
                    patient_out = self.patient_mlp(patient_data[patient_idx])
                
                # Ensure proper dimensions
                patient_out = patient_out.unsqueeze(0) if patient_out.dim() == 1 else patient_out
                
            except Exception as e:
                print(f"Error processing patient data: {str(e)}")
                # Create a zero tensor as fallback
                patient_out = torch.zeros(1, self.patient_mlp[-1].out_features, device=struct_feat.device)

            # Concatenate features based on gene node usage
            if self.use_gene_nodes and gene_feat is not None:
                patient_features = torch.cat([
                    struct_feat,
                    gene_feat,
                    combined_feat,
                    patient_out
                ], dim=1)
            else:
                patient_features = torch.cat([
                    struct_feat,
                    patient_out
                ], dim=1)

            return patient_features

        except Exception as e:
            print(f"Error processing patient features: {str(e)}")
            traceback.print_exc()
            return None

    def forward(self, data, patient_data):
        """Forward pass with gene control flags."""
        try:
            if not hasattr(data, 'node_types') or not hasattr(data, 'edge_types'):
                return None, None, None

            if 'structure' not in data.node_types:
                return None, None, None

            if self.use_gene_nodes and 'gene' not in data.node_types:
                return None, None, None

            # Process each patient
            patient_features = []
            for patient_id in list(dict.fromkeys(data.patient_ids)):
                patient_indices = [i for i, pid in enumerate(data.patient_ids) if pid == patient_id]
                visit_indices = range(len(patient_indices))
                
                if patient_indices:
                    features = self._process_patient_features(
                        data, patient_data, patient_id, patient_indices, visit_indices
                    )
                    if features is not None:
                        features = features.squeeze(0)
                        patient_features.append(features)

            if not patient_features:
                return None, None, None

            # Stack and process features
            patient_features = torch.stack(patient_features)
            shared_features = self.feature_aggregator(patient_features)

            # Get predictions
            deephit_preds = self.deephit_net(shared_features)
            deephit_preds = F.softmax(deephit_preds, dim=1)  # Apply softmax for probability distribution
            
            # PACC predictions with clamping
            paccv6_preds = self.paccv6_head(shared_features)
            paccv6_preds = torch.clamp(paccv6_preds, min=0, max=100)

            return shared_features, deephit_preds, paccv6_preds.squeeze(-1)

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            return None, None, None

    def get_prediction_for_patient(self, patient_id, data, patient_data):
        """Get predictions for a specific patient."""
        try:
            # Find all visits for this patient
            patient_indices = [i for i, pid in enumerate(data.patient_ids) if pid == patient_id]
            
            if not patient_indices:
                print(f"No data found for patient {patient_id}")
                return None
            
            visit_indices = range(len(patient_indices))
            features = self._process_patient_features(data, patient_data, patient_id, patient_indices, visit_indices)
            
            if features is None:
                return None
            
            # Process features through the prediction heads
            shared_features = self.feature_aggregator(features)
            deephit_preds = self.deephit_net(shared_features)
            deephit_preds = F.softmax(deephit_preds, dim=1)  # Apply softmax
            
            paccv6_preds = self.paccv6_head(shared_features)
            paccv6_preds = torch.clamp(paccv6_preds, min=0, max=100)
            
            return {
                'deephit_predictions': deephit_preds.squeeze().detach().cpu().numpy(),
                'paccv6_prediction': paccv6_preds.squeeze().detach().cpu().numpy(),
                'shared_features': shared_features.squeeze().detach().cpu().numpy()
            }
            
        except Exception as e:
            print(f"Error in patient prediction: {str(e)}")
            traceback.print_exc()
            return None

    def get_feature_importance(self, data, patient_data):
        """Calculate feature importance scores."""
        try:
            self.eval()
            with torch.no_grad():
                baseline_features, baseline_deephit, baseline_paccv6 = self.forward(data, patient_data)
                
                if baseline_features is None:
                    return None
                
                importance_scores = {}
                
                # Structure importance
                structure_importance = []
                for i in range(len(self.structures)):
                    temp_data = copy.deepcopy(data)
                    temp_data['structure'].x[:, i] = 0
                    perturbed_features, _, _ = self.forward(temp_data, patient_data)
                    if perturbed_features is not None:
                        diff = torch.mean(torch.abs(baseline_features - perturbed_features))
                        structure_importance.append(diff.item())
                    else:
                        structure_importance.append(0)
                importance_scores['structure'] = structure_importance
                
                # Gene importance if enabled
                if self.use_gene_nodes and 'gene' in data.node_types:
                    gene_importance = []
                    gene_features = data['gene'].x
                    for i in range(gene_features.size(1)):
                        temp_data = copy.deepcopy(data)
                        temp_data['gene'].x[:, i] = 0
                        perturbed_features, _, _ = self.forward(temp_data, patient_data)
                        if perturbed_features is not None:
                            diff = torch.mean(torch.abs(baseline_features - perturbed_features))
                            gene_importance.append(diff.item())
                        else:
                            gene_importance.append(0)
                    importance_scores['gene'] = gene_importance
                
                return importance_scores
                
        except Exception as e:
            print(f"Error in feature importance calculation: {str(e)}")
            traceback.print_exc()
            return None