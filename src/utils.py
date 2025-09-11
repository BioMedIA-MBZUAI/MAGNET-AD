"""
MAGNET-AD Utility Functions
Provides data loading, processing, and evaluation utilities.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import traceback
from torch_geometric.data import HeteroData
from typing import List, Dict, Union, Optional, Tuple


class PatientDataset(Dataset):
    """Dataset for patient clinical data from CSV"""
    
    def __init__(self, data, patient_ids=None, device=None):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame or numpy array with patient data
            patient_ids: List of patient IDs corresponding to data rows
            device: Device to place tensors on
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.select_dtypes(include=[np.number]).values
        else:
            self.data = data

        self.patient_ids = patient_ids
        self.device = device

        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Data must be a numpy array or pandas DataFrame, got {type(self.data)}")

        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get tensor for a patient"""
        if isinstance(idx, slice):
            # Handle slice indexing
            data = self.data[idx]
            tensors = [torch.from_numpy(d).float() for d in data]
            if self.device:
                tensors = [t.to(self.device) for t in tensors]
            return torch.stack(tensors) if tensors else torch.tensor([], dtype=torch.float32)
        else:
            # Handle single index
            tensor = torch.from_numpy(self.data[idx]).float()
            if self.device:
                tensor = tensor.to(self.device)
            return tensor


def load_csv_data(ids: List[str], csv_file: str) -> pd.DataFrame:
    """
    Load and preprocess CSV data for specified patient IDs.
    
    Args:
        ids: List of patient IDs to load
        csv_file: Path to the CSV file
        
    Returns:
        DataFrame with preprocessed data
    """
    try:
        df = pd.read_csv(csv_file)

        # Ensure BID column exists
        if 'BID' not in df.columns:
            raise ValueError("CSV missing required 'BID' column")

        # Drop duplicate BID rows by keeping the first occurrence
        df = df.drop_duplicates(subset=['BID'], keep='first')

        # De-duplicate input ids while preserving order
        seen = set()
        unique_ids = []
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)

        # Select only rows for given IDs
        df = df[df['BID'].isin(unique_ids)]

        # Set BID as index
        df.set_index('BID', inplace=True)

        # Intersect reindex target with available index to avoid missing duplicates confusion
        # Reorder to match input order; missing IDs will yield NaNs which we handle below
        df = df.reindex(unique_ids)

        # Ensure all remaining columns are numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]

        # Convert to float32
        df = df.astype(np.float32)

        # Fill NaN values with column means (including those from missing IDs)
        df = df.fillna(df.mean())

        return df
        
    except Exception as e:
        print(f"Error loading CSV data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def collate_hetero_data(batch: List[HeteroData]) -> HeteroData:
    """
    Collate function for heterogeneous graph data.
    
    Args:
        batch: List of HeteroData objects
        
    Returns:
        Batched HeteroData object
    """
    if not batch:
        return None

    try:
        batched_data = HeteroData()
        first_data = batch[0]

        # Handle structure node features (concatenate across all visits for all patients)
        structure_features = []
        total_structures = 0

        for data in batch:
            if not hasattr(data['structure'], 'x'):
                print(f"Warning: Missing structure features for patient {data.patient_id}")
                continue
            structure_features.append(data['structure'].x)
            total_structures += data['structure'].x.size(0)

        if structure_features:
            batched_data['structure'].x = torch.cat(structure_features, dim=0)
        else:
            print("Error: No valid structure features found")
            return None

        # Handle gene features (same for all patients)
        if hasattr(first_data, 'gene') and hasattr(first_data['gene'], 'x'):
            batched_data['gene'].x = first_data['gene'].x

        # Handle all structure-to-structure edge types generically, preserving relation names
        structure_offset = 0
        # Initialize containers per edge type
        ss_edges: Dict[tuple, Dict[str, torch.Tensor]] = {}

        for data in batch:
            for edge_type in data.edge_types:
                if edge_type[0] == 'structure' and edge_type[2] == 'structure':
                    edge_index = data[edge_type].edge_index.clone()
                    edge_attr = getattr(data[edge_type], 'edge_attr', None)
                    # Apply structure offset to both endpoints
                    edge_index = edge_index + structure_offset
                    if edge_type not in ss_edges:
                        ss_edges[edge_type] = {
                            'edge_index_list': [edge_index],
                            'edge_attr_list': [edge_attr] if edge_attr is not None else []
                        }
                    else:
                        ss_edges[edge_type]['edge_index_list'].append(edge_index)
                        if edge_attr is not None:
                            ss_edges[edge_type]['edge_attr_list'].append(edge_attr)

            structure_offset += data['structure'].x.size(0)

        # Concatenate per edge type
        for edge_type, parts in ss_edges.items():
            batched_data[edge_type].edge_index = torch.cat(parts['edge_index_list'], dim=1)
            if parts['edge_attr_list']:
                batched_data[edge_type].edge_attr = torch.cat(parts['edge_attr_list'], dim=0)

        # Handle gene-related edges
        if hasattr(first_data, 'gene'):
            # Gene-gene co-expression edges
            if ('gene', 'coexpressed_with', 'gene') in first_data.edge_types:
                batched_data['gene', 'coexpressed_with', 'gene'].edge_index = first_data['gene', 'coexpressed_with', 'gene'].edge_index
                batched_data['gene', 'coexpressed_with', 'gene'].edge_attr = first_data['gene', 'coexpressed_with', 'gene'].edge_attr

            # Gene-structure edges
            gene_struct_indices = []
            gene_struct_attrs = []
            structure_offset = 0

            for data in batch:
                if ('gene', 'expressed_in', 'structure') in data.edge_types:
                    edge_index = data['gene', 'expressed_in', 'structure'].edge_index.clone()
                    edge_index[1] = edge_index[1] + structure_offset  # Offset structure indices
                    gene_struct_indices.append(edge_index)
                    gene_struct_attrs.append(data['gene', 'expressed_in', 'structure'].edge_attr)

                structure_offset += data['structure'].x.size(0)

            if gene_struct_indices:
                batched_data['gene', 'expressed_in', 'structure'].edge_index = torch.cat(gene_struct_indices, dim=1)
                batched_data['gene', 'expressed_in', 'structure'].edge_attr = torch.cat(gene_struct_attrs, dim=0)

        # Handle metadata (one per patient)
        patient_ids = []
        patient_visits = []
        survival_times_list = []
        events_list = []
        pacc_scores_list = []

        for data in batch:
            patient_ids.append(data.patient_id)
            
            # Store visit information
            if hasattr(data, 'visits') and data.visits:
                patient_visits.append(data.visits)
            else:
                patient_visits.append([0])  # Single visit fallback
            
            # Store survival data (use first visit or last visit)
            if hasattr(data, 'survival_times') and len(data.survival_times) > 0:
                survival_times_list.append(data.survival_times[-1].item())  # Use last visit
            else:
                survival_times_list.append(0.0)
                
            if hasattr(data, 'events') and len(data.events) > 0:
                events_list.append(data.events[-1].item())  # Use last visit
            else:
                events_list.append(0.0)
                
            if hasattr(data, 'pacc_scores') and len(data.pacc_scores) > 0:
                pacc_scores_list.append(data.pacc_scores[-1].item())  # Use last visit
            else:
                pacc_scores_list.append(0.0)

        # Store metadata in batched data
        batched_data.patient_ids = patient_ids
        batched_data.patient_visits = patient_visits
        batched_data.survival_times = torch.tensor(survival_times_list)
        batched_data.events = torch.tensor(events_list)
        batched_data.pacc_scores = torch.tensor(pacc_scores_list)

        return batched_data

    except Exception as e:
        print(f"Error in collate function: {str(e)}")
        traceback.print_exc()
        return None


def save_prediction_results(results, output_file):
    """Save prediction results to a file"""
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        traceback.print_exc()


def load_prediction_results(input_file):
    """Load prediction results from a file"""
    try:
        with open(input_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        traceback.print_exc()
        return {}


def calculate_model_statistics(model):
    """Calculate model parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    return stats


def create_train_val_test_split(patient_ids: List[str], train_ratio: float = 0.7, 
                               val_ratio: float = 0.15, random_seed: int = 42) -> Dict[str, List[str]]:
    """Create train/validation/test splits"""
    np.random.seed(random_seed)
    
    # Shuffle patient IDs
    shuffled_ids = np.array(patient_ids)
    np.random.shuffle(shuffled_ids)
    
    # Calculate split sizes
    n_patients = len(shuffled_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Create splits
    train_ids = shuffled_ids[:n_train].tolist()
    val_ids = shuffled_ids[n_train:n_train + n_val].tolist()
    test_ids = shuffled_ids[n_train + n_val:].tolist()
    
    return {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }


def normalize_features(features: torch.Tensor, method: str = 'zscore') -> torch.Tensor:
    """Normalize features using specified method"""
    if method == 'zscore':
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_vals = torch.min(features, dim=0, keepdim=True)[0]
        max_vals = torch.max(features, dim=0, keepdim=True)[0]
        return (features - min_vals) / (max_vals - min_vals + 1e-8)
    else:
        return features


def compute_graph_statistics(graphs: List[HeteroData]) -> Dict[str, float]:
    """Compute statistics about the graphs"""
    if not graphs:
        return {}
    
    stats = {
        'num_graphs': len(graphs),
        'avg_structure_nodes': 0,
        'avg_gene_nodes': 0,
        'avg_structure_edges': 0,
        'avg_gene_edges': 0,
        'avg_gene_structure_edges': 0,
        'avg_visits_per_patient': 0
    }
    
    structure_nodes = []
    gene_nodes = []
    structure_edges = []
    gene_edges = []
    gene_structure_edges = []
    visits = []
    
    for graph in graphs:
        if 'structure' in graph.node_types:
            structure_nodes.append(graph['structure'].x.size(0))
        
        if 'gene' in graph.node_types:
            gene_nodes.append(graph['gene'].x.size(0))
        
        if ('structure', 'bold_correlated', 'structure') in graph.edge_types:
            structure_edges.append(graph['structure', 'bold_correlated', 'structure'].edge_index.size(1))
        
        if ('gene', 'coexpressed_with', 'gene') in graph.edge_types:
            gene_edges.append(graph['gene', 'coexpressed_with', 'gene'].edge_index.size(1))
        
        if ('gene', 'expressed_in', 'structure') in graph.edge_types:
            gene_structure_edges.append(graph['gene', 'expressed_in', 'structure'].edge_index.size(1))
        
        if hasattr(graph, 'visits'):
            visits.append(len(graph.visits))
    
    if structure_nodes:
        stats['avg_structure_nodes'] = np.mean(structure_nodes)
    if gene_nodes:
        stats['avg_gene_nodes'] = np.mean(gene_nodes)
    if structure_edges:
        stats['avg_structure_edges'] = np.mean(structure_edges)
    if gene_edges:
        stats['avg_gene_edges'] = np.mean(gene_edges)
    if gene_structure_edges:
        stats['avg_gene_structure_edges'] = np.mean(gene_structure_edges)
    if visits:
        stats['avg_visits_per_patient'] = np.mean(visits)
    
    return stats
