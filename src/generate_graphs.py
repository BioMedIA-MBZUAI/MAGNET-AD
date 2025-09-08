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

# Import the BrainGraphBuilder class
from brain_graph_builder import BrainGraphBuilder


def split_patients(patient_ids, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Split patient IDs into train/val/test sets.
    
    Args:
        patient_ids: List of patient IDs
        train_ratio: Ratio of patients for training
        val_ratio: Ratio of patients for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with train_ids, val_ids, and test_ids
    """
    random.seed(random_seed)
    
    # Shuffle patient IDs
    shuffled_ids = patient_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split indices
    n_patients = len(shuffled_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patients
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    return {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }


def generate_graphs(args):
    """
    Generate brain graphs for all patients.
    
    Args:
        args: Command line arguments
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
        splits = split_patients(
            patient_ids, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.random_seed
        )
        
        # Save splits
        with open(output_dir / "splits.pkl", 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"Splits saved: {len(splits['train_ids'])} train, "
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
        print(f"\nProcessing BOLD threshold: {threshold}")
        
        # Initialize graph builder
        graph_builder = BrainGraphBuilder(
            embeddings_base_path=Path(args.embeddings_dir),
            correlation_path=Path(args.correlation_path),
            radiomics_path=Path(args.radiomics_path) if args.radiomics_path else None,
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
                
            print(f"Processing {len(split_ids)} patients for {split_name} split")
            split_graphs = {}
            
            for patient_id in tqdm(split_ids):
                try:
                    # Build patient graph
                    patient_graph = graph_builder.build_patient_graph(patient_id)
                    
                    # Store graph
                    split_graphs[patient_id] = patient_graph
                    
                except Exception as e:
                    if args.verbose:
                        print(f"Error processing patient {patient_id}: {str(e)}")
                        traceback.print_exc()
                    else:
                        print(f"Error processing patient {patient_id}")
            
            # Save graphs for this split
            output_path = threshold_dir / f"{split_name}_graphs_bold{threshold}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(split_graphs, f)
            
            print(f"Saved {len(split_graphs)} graphs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate brain graphs for BrainTemporalGNN')
    
    # Input data paths
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing brain structure embeddings')
    parser.add_argument('--correlation_path', type=str, required=True,
                        help='Path to BOLD correlations CSV')
    parser.add_argument('--clinical_data', type=str, required=True,
                        help='Path to clinical data CSV')
    parser.add_argument('--gene_gene_path', type=str, required=True,
                        help='Path to gene-gene interactions CSV')
    parser.add_argument('--gene_structure_path', type=str, required=True,
                        help='Path to gene-structure connections CSV')
    parser.add_argument('--gene_embeddings_dir', type=str, required=True,
                        help='Directory containing gene embeddings')
    parser.add_argument('--radiomics_path', type=str, default=None,
                        help='Path to radiomics CSV (optional)')
    
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
    generate_graphs(args)