"""
MAGNET-AD Graph Generation Script
Generates heterogeneous brain graphs following the exact paper description.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import traceback
from tqdm import tqdm
import random

from graph_builder import GraphBuilder
from utils import create_train_val_test_split, compute_graph_statistics


def generate_graphs(args):
    """
    Generate brain graphs for all patients following MAGNET-AD paper.
    
    Creates heterogeneous graphs with:
    - 32 brain structure nodes with 107-dimensional radiomic features
    - 100 AD-associated gene nodes with 768-dimensional embeddings
    - Three types of edges: structure-structure, gene-gene, gene-structure
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up BOLD threshold directories
    bold_dirs = {}
    for threshold in args.bold_thresholds:
        threshold_dir = output_dir / f"bold_{threshold}"
        threshold_dir.mkdir(exist_ok=True)
        bold_dirs[threshold] = threshold_dir
    
    # Load clinical data to get patient IDs
    print(f"Loading clinical data from {args.clinical_data}")
    clinical_df = pd.read_csv(args.clinical_data)
    all_patient_ids = clinical_df['BID'].unique().tolist()
    
    # Filter patients if list provided
    if args.patient_ids:
        patient_ids = args.patient_ids.split(',')
        # Check if all specified patients exist in clinical data
        missing = [pid for pid in patient_ids if pid not in all_patient_ids]
        if missing:
            print(f"Warning: {len(missing)} patient IDs not found in clinical data")
        
        # Keep only existing patients
        patient_ids = [pid for pid in patient_ids if pid in all_patient_ids]
    else:
        patient_ids = all_patient_ids
    
    print(f"Generating graphs for {len(patient_ids)} patients")
    
    # Create splits
    if args.create_splits:
        print("Creating train/val/test splits")
        splits = create_train_val_test_split(
            patient_ids, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.random_seed
        )
        
        # Save splits
        with open(output_dir / "splits.pkl", 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"Splits created: {len(splits['train_ids'])} train, "
              f"{len(splits['val_ids'])} val, {len(splits['test_ids'])} test")
    else:
        # Use pre-existing splits or put all patients in test
        if os.path.exists(output_dir / "splits.pkl"):
            with open(output_dir / "splits.pkl", 'rb') as f:
                splits = pickle.load(f)
                print(f"Loaded existing splits: {len(splits['train_ids'])} train, "
                      f"{len(splits['val_ids'])} val, {len(splits['test_ids'])} test")
        else:
            # Default: put all in test split
            splits = {
                'train_ids': [],
                'val_ids': [],
                'test_ids': patient_ids
            }
            
            # Save the default splits
            with open(output_dir / "splits.pkl", 'wb') as f:
                pickle.dump(splits, f)
            
            print(f"Created default splits with all {len(patient_ids)} patients in test set")
    
    # Process each BOLD threshold
    for threshold in args.bold_thresholds:
        threshold_dir = bold_dirs[threshold]
        print(f"\nProcessing BOLD correlation threshold: {threshold}%")
        
        # Initialize the GraphBuilder following paper specifications
        graph_builder = GraphBuilder(
            embeddings_base_path=Path(args.embeddings_dir),
            correlation_path=Path(args.correlation_path),
            radiomics_path=Path(args.radiomics_path),
            clinical_data_path=Path(args.clinical_data),
            gene_gene_path=Path(args.gene_gene_path),
            gene_structure_path=Path(args.gene_structure_path),
            gene_embeddings_path=Path(args.gene_embeddings_dir),
            bold_threshold=threshold,
            verbose=args.verbose
        )
        
        # Process patients by split
        for split_name, split_ids in splits.items():
            if not split_ids:
                continue
                
            print(f"\nProcessing {len(split_ids)} patients for {split_name} split")
            split_graphs = {}
            successful_graphs = 0
            failed_patients = []
            
            for patient_id in tqdm(split_ids, desc=f"Building {split_name} graphs"):
                try:
                    # Build heterogeneous graph following paper description
                    patient_graph = graph_builder.build_patient_graph(patient_id)
                    
                    # Validate graph structure
                    if validate_graph_structure(patient_graph):
                        split_graphs[patient_id] = patient_graph
                        successful_graphs += 1
                    else:
                        failed_patients.append(patient_id)
                        if args.verbose:
                            print(f"Invalid graph structure for patient {patient_id}")
                    
                except Exception as e:
                    failed_patients.append(patient_id)
                    if args.verbose:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        traceback.print_exc()
                    else:
                        print(f"Error processing patient {patient_id}")
            
            # Save graphs for this split
            if split_graphs:
                output_path = threshold_dir / f"{split_name}_graphs_bold{threshold}.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(split_graphs, f)
                
                print(f"Saved {len(split_graphs)} graphs to {output_path}")
                
                # Compute and save graph statistics
                if args.save_stats:
                    stats = compute_graph_statistics(list(split_graphs.values()))
                    stats_path = threshold_dir / f"{split_name}_stats_bold{threshold}.json"
                    import json
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"Graph statistics saved to {stats_path}")
            
            if failed_patients:
                print(f"Failed to process {len(failed_patients)} patients: {failed_patients[:5]}{'...' if len(failed_patients) > 5 else ''}")


def validate_graph_structure(graph):
    """
    Validate that the graph follows the MAGNET-AD paper structure.
    
    Expected structure:
    - Structure nodes: 32 brain regions × number of visits
    - Gene nodes: 100 AD-associated genes
    - Edges: BOLD correlations, gene co-expression, gene-structure connections
    """
    if graph is None:
        return False
    
    try:
        # Check node types
        if 'structure' not in graph.node_types:
            return False
        
        # Check structure nodes (should be multiple of 32 for multiple visits)
        num_structure_nodes = graph['structure'].x.size(0)
        if num_structure_nodes == 0 or num_structure_nodes % 32 != 0:
            return False
        
        # Check structure feature dimensions (should be 512 after projection)
        if graph['structure'].x.size(1) != 512:
            return False
        
        # Check gene nodes if present
        if 'gene' in graph.node_types:
            num_gene_nodes = graph['gene'].x.size(0)
            if num_gene_nodes > 0:
                # Gene features should be 512-dimensional after projection
                if graph['gene'].x.size(1) != 512:
                    return False
        
        # Check required metadata
        required_attrs = ['patient_id', 'visits', 'survival_times', 'events', 'pacc_scores']
        for attr in required_attrs:
            if not hasattr(graph, attr):
                return False
        
        # Check visits consistency
        num_visits = len(graph.visits)
        if num_visits == 0:
            return False
        
        # Structure nodes should match visits (32 regions per visit)
        expected_structure_nodes = 32 * num_visits
        if num_structure_nodes != expected_structure_nodes:
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating graph structure: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MAGNET-AD heterogeneous brain graphs')
    
    # Input data paths
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing brain structure embeddings')
    parser.add_argument('--correlation_path', type=str, required=True,
                        help='Path to BOLD correlations CSV')
    parser.add_argument('--clinical_data', type=str, required=True,
                        help='Path to clinical data CSV')
    parser.add_argument('--gene_gene_path', type=str, required=True,
                        help='Path to gene-gene co-expression CSV')
    parser.add_argument('--gene_structure_path', type=str, required=True,
                        help='Path to gene-structure mRNA expression CSV')
    parser.add_argument('--gene_embeddings_dir', type=str, required=True,
                        help='Directory containing gene embeddings')
    parser.add_argument('--radiomics_path', type=str, required=True,
                        help='Path to radiomic features CSV (107-dimensional)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./brain_graphs',
                        help='Directory to save generated graphs')
    
    # Processing options
    parser.add_argument('--bold_thresholds', type=int, nargs='+', default=[50],
                        help='BOLD correlation thresholds to generate (e.g., 50 70 90)')
    parser.add_argument('--patient_ids', type=str, default=None,
                        help='Comma-separated list of patient IDs (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save graph statistics')
    
    # Split options
    parser.add_argument('--create_splits', action='store_true',
                        help='Create new train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of patients for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of patients for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for splitting')
    
    args = parser.parse_args()
    
    print("MAGNET-AD Graph Generation")
    print("=" * 50)
    print(f"Following paper specification:")
    print(f"- 32 brain structure nodes with 107-dim radiomic features")
    print(f"- 100 AD-associated gene nodes with 768-dim embeddings")
    print(f"- BOLD correlation, gene co-expression, and gene-structure edges")
    print(f"- Temporal dynamics through multiple visit connections")
    print("=" * 50)
    
    generate_graphs(args)
