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
    """Dataset for patient clinical data from CSV."""
    
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
        """Get tensor for a patient."""
        if isinstance(idx, slice):
            # Handle slice indexing
            data = self.data[idx]
            tensors = [torch.from_numpy(d).float() for d in data]
            if self.device:
                tensors = [t.to(self.device) for t in tensors]
            return torch.stack(tensors) if tensors else torch.tensor([])
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

        # Select only rows for given IDs
        df = df[df['BID'].isin(ids)]

        # Set BID as index
        df.set_index('BID', inplace=True)

        # Ensure all remaining columns are numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]

        # Convert to float32
        df = df.astype(np.float32)

        return df
        
    except Exception as e:
        print(f"Error loading CSV data: {str(e)}")
        traceback.print_exc()
        # Return empty DataFrame with same structure
        return pd.DataFrame(columns=['BID'] + list(numeric_cols))


def load_patient_graph(patient_id: str, data_dir: str, threshold: int) -> Optional[HeteroData]:
    """
    Load graph data for a single patient.
    
    Args:
        patient_id: Patient ID
        data_dir: Directory containing graph data
        threshold: BOLD threshold value
        
    Returns:
        HeteroData graph for the patient or None if not found
    """
    try:
        # Determine which split the patient belongs to
        splits_path = Path(data_dir) / 'splits.pkl'
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
            
        split_name = None
        for split in ['train_ids', 'val_ids', 'test_ids']:
            if patient_id in splits[split]:
                split_name = split.split('_')[0]  # 'train', 'val', or 'test'
                break
                
        if not split_name:
            print(f"Warning: Patient {patient_id} not found in any split")
            return None
            
        # Load the graph
        threshold_dir = Path(data_dir) / f'bold_{threshold}'
        graph_file = threshold_dir / f'{split_name}_graphs_bold{threshold}.pkl'
        
        with open(graph_file, 'rb') as f:
            graphs_dict = pickle.load(f)
            
        if patient_id in graphs_dict:
            return graphs_dict[patient_id]
        else:
            print(f"Warning: Patient {patient_id} not found in {graph_file}")
            return None
            
    except Exception as e:
        print(f"Error loading graph for patient {patient_id}: {str(e)}")
        traceback.print_exc()
        return None


def setup_patient_graphs(
    patient_ids: List[str], 
    data_dir: str, 
    threshold: int,
    use_gene_edges: bool = True,
    use_gene_nodes: bool = True
) -> Dict[str, HeteroData]:
    """
    Set up graph data for multiple patients.
    
    Args:
        patient_ids: List of patient IDs
        data_dir: Directory containing graph data
        threshold: BOLD threshold value
        use_gene_edges: Whether to use gene-gene edges
        use_gene_nodes: Whether to use gene nodes
        
    Returns:
        Dictionary of patient graphs keyed by patient ID
    """
    patient_graphs = {}
    
    # Load splits
    try:
        splits_path = Path(data_dir) / 'splits.pkl'
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
            
        # Determine which split each patient belongs to
        split_mapping = {}
        for patient_id in patient_ids:
            for split in ['train_ids', 'val_ids', 'test_ids']:
                if patient_id in splits[split]:
                    split_mapping[patient_id] = split.split('_')[0]  # 'train', 'val', or 'test'
                    break
        
        # Load graphs by split for efficiency
        for split_name in set(split_mapping.values()):
            split_patients = [pid for pid, split in split_mapping.items() if split == split_name]
            if not split_patients:
                continue
                
            threshold_dir = Path(data_dir) / f'bold_{threshold}'
            graph_file = threshold_dir / f'{split_name}_graphs_bold{threshold}.pkl'
            
            if not graph_file.exists():
                print(f"Warning: Graph file {graph_file} not found")
                continue
                
            with open(graph_file, 'rb') as f:
                graphs_dict = pickle.load(f)
                
            # Add to patient_graphs
            for pid in split_patients:
                if pid in graphs_dict:
                    # If not using gene nodes/edges, filter them out to save memory
                    graph = graphs_dict[pid]
                    if not use_gene_nodes and 'gene' in graph.node_types:
                        # Create a new graph without gene nodes
                        filtered_graph = HeteroData()
                        filtered_graph['structure'] = graph['structure']
                        
                        # Copy structure edges
                        if ('structure', 'bold_correlated', 'structure') in graph.edge_types:
                            filtered_graph['structure', 'bold_correlated', 'structure'] = \
                                graph['structure', 'bold_correlated', 'structure']
                                
                        if ('structure', 'temporally_connected', 'structure') in graph.edge_types:
                            filtered_graph['structure', 'temporally_connected', 'structure'] = \
                                graph['structure', 'temporally_connected', 'structure']
                        
                        # Copy metadata
                        for attr in ['patient_id', 'visits', 'survival_times', 'events', 'paccv6_scores']:
                            if hasattr(graph, attr):
                                setattr(filtered_graph, attr, getattr(graph, attr))
                                
                        patient_graphs[pid] = filtered_graph
                    else:
                        patient_graphs[pid] = graph
        
        print(f"Loaded graphs for {len(patient_graphs)} patients")
        return patient_graphs
        
    except Exception as e:
        print(f"Error setting up patient graphs: {str(e)}")
        traceback.print_exc()
        return patient_graphs


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

        # Handle structure node features
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

        # Only include gene features if present in the first data object
        if hasattr(first_data, 'gene') and hasattr(first_data['gene'], 'x'):
            batched_data['gene'].x = first_data['gene'].x

        # Handle BOLD correlation edges
        spatial_edge_indices = []
        spatial_edge_attrs = []
        structure_offset = 0

        for data in batch:
            if ('structure', 'bold_correlated', 'structure') in data.edge_types:
                edge_index = data['structure', 'bold_correlated', 'structure'].edge_index.clone()
                edge_index = edge_index + structure_offset
                spatial_edge_indices.append(edge_index)
                spatial_edge_attrs.append(data['structure', 'bold_correlated', 'structure'].edge_attr)

            structure_offset += data['structure'].x.size(0)

        if spatial_edge_indices:
            batched_data['structure', 'bold_correlated', 'structure'].edge_index = torch.cat(spatial_edge_indices, dim=1)
            batched_data['structure', 'bold_correlated', 'structure'].edge_attr = torch.cat(spatial_edge_attrs, dim=0)

        # Handle temporal edges
        temporal_edge_indices = []
        temporal_edge_attrs = []
        structure_offset = 0

        for data in batch:
            if ('structure', 'temporally_connected', 'structure') in data.edge_types:
                edge_index = data['structure', 'temporally_connected', 'structure'].edge_index.clone()
                edge_index = edge_index + structure_offset
                temporal_edge_indices.append(edge_index)
                temporal_edge_attrs.append(data['structure', 'temporally_connected', 'structure'].edge_attr)

            structure_offset += data['structure'].x.size(0)

        if temporal_edge_indices:
            batched_data['structure', 'temporally_connected', 'structure'].edge_index = torch.cat(temporal_edge_indices, dim=1)
            batched_data['structure', 'temporally_connected', 'structure'].edge_attr = torch.cat(temporal_edge_attrs, dim=0)

        # Handle gene-related edges for each stage if gene exists in first item
        if hasattr(first_data, 'gene'):
            for stage in ['early', 'mid', 'late']:
                # Gene-gene interactions
                gene_edge_indices = []
                gene_edge_attrs = []

                for data in batch:
                    edge_type = ('gene', f'interacts_{stage}', 'gene')
                    if edge_type in data.edge_types:
                        gene_edge_indices.append(data[edge_type].edge_index)
                        gene_edge_attrs.append(data[edge_type].edge_attr)

                if gene_edge_indices:
                    batched_data['gene', f'interacts_{stage}', 'gene'].edge_index = gene_edge_indices[0]
                    batched_data['gene', f'interacts_{stage}', 'gene'].edge_attr = gene_edge_attrs[0]

                # Gene-structure connections
                gene_struct_indices = []
                gene_struct_attrs = []
                structure_offset = 0

                for data in batch:
                    edge_type = ('gene', f'connects_{stage}', 'structure')
                    if edge_type in data.edge_types:
                        edge_index = data[edge_type].edge_index.clone()
                        edge_index[1] = edge_index[1] + structure_offset
                        gene_struct_indices.append(edge_index)
                        gene_struct_attrs.append(data[edge_type].edge_attr)

                    structure_offset += data['structure'].x.size(0)

                if gene_struct_indices:
                    batched_data['gene', f'connects_{stage}', 'structure'].edge_index = torch.cat(gene_struct_indices, dim=1)
                    batched_data['gene', f'connects_{stage}', 'structure'].edge_attr = torch.cat(gene_struct_attrs, dim=0)

        # Handle metadata and survival information
        patient_ids = []
        survival_times_list = []
        events_list = []
        paccv6_scores_list = []

        for data in batch:
            patient_ids.append(data.patient_id)
            survival_times_list.append(data.survival_times[-1] if hasattr(data, 'survival_times') else torch.tensor(0.0))
            events_list.append(data.events[-1] if hasattr(data, 'events') else torch.tensor(0.0))
            paccv6_scores_list.append(data.paccv6_scores[-1] if hasattr(data, 'paccv6_scores') else torch.tensor(0.0))

        # Store metadata in batched data
        batched_data.patient_ids = patient_ids
        batched_data.survival_times = torch.stack(survival_times_list)
        batched_data.events = torch.stack(events_list)
        batched_data.paccv6_scores = torch.stack(paccv6_scores_list)

        return batched_data

    except Exception as e:
        print(f"Error in collate function: {str(e)}")
        traceback.print_exc()
        return None


def save_prediction_results(results, output_file):
    """
    Save prediction results to a file.
    
    Args:
        results: Dictionary of prediction results
        output_file: Path to save the results
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        traceback.print_exc()


def load_prediction_results(input_file):
    """
    Load prediction results from a file.
    
    Args:
        input_file: Path to the results file
        
    Returns:
        Dictionary of prediction results
    """
    try:
        with open(input_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        traceback.print_exc()
        return {}