import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import traceback
from tqdm import tqdm
import time

# Import required modules
from brain_graph_builder import BrainGraphBuilder
from model import BrainTemporalGNN
from utils import PatientDataset


def load_model(model_path, device=None):
    """
    Load a trained BrainTemporalGNN model.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        tuple: (model, config) - The loaded model and its configuration
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract configuration
        config = {
            'bold_threshold': checkpoint.get('bold_threshold', 50),
            'use_gene_edges': checkpoint.get('use_gene_edges', True),
            'use_gene_nodes': checkpoint.get('use_gene_nodes', True)
        }
        
        # Determine model parameters from state dict
        state_dict = checkpoint['model_state_dict']
        
        # Infer csv_input_dim from patient_mlp.0.weight shape
        if 'patient_mlp.0.weight' in state_dict:
            csv_input_dim = state_dict['patient_mlp.0.weight'].shape[1]
        else:
            csv_input_dim = 100  # Default fallback
            
        # Create model with extracted parameters
        model = BrainTemporalGNN(
            in_channels=512,
            hidden_channels=512,
            num_blocks=3,
            csv_input_dim=csv_input_dim,
            csv_hidden_dim=64,
            csv_output_dim=32,
            dropout=0.2,
            use_gene_edges=config['use_gene_edges'],
            use_gene_nodes=config['use_gene_nodes']
        ).to(device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, config
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        raise


def process_patient(patient_id, graph_builder, model, patient_data_tensor, device):
    """
    Process a single patient through graph building and inference.
    
    Args:
        patient_id: Patient ID
        graph_builder: BrainGraphBuilder instance
        model: Trained model
        patient_data_tensor: Patient's feature tensor
        device: Computation device
        
    Returns:
        dict: Prediction results
    """
    try:
        # Build patient graph
        patient_graph = graph_builder.build_patient_graph(patient_id)
        patient_graph = patient_graph.to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model.get_prediction_for_patient(
                patient_id, 
                patient_graph, 
                patient_data_tensor
            )
        
        if prediction is None:
            return None
            
        # Calculate survival probabilities from DeepHit predictions
        deephit_preds = prediction['deephit_predictions']
        surv_probs = 1 - np.cumsum(deephit_preds)
        
        # Calculate risk score
        risk_score = -surv_probs[len(surv_probs)//2]
        
        # Package results
        result = {
            'patient_id': patient_id,
            'paccv6_prediction': float(prediction['paccv6_prediction']),
            'risk_score': float(risk_score),
            'survival_probabilities': surv_probs.tolist(),
            'num_visits': len(patient_graph.visits) if hasattr(patient_graph, 'visits') else 1
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        traceback.print_exc()
        return None


def run_workflow(args):
    """
    Run the complete workflow: graph generation and inference.
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load model
    print(f"Loading model from {args.model_path}")
    model, config = load_model(args.model_path, device)
    
    # Step 2: Initialize graph builder
    print("Initializing graph builder")
    graph_builder = BrainGraphBuilder(
        embeddings_base_path=Path(args.embeddings_dir),
        correlation_path=Path(args.correlation_path),
        radiomics_path=Path(args.radiomics_path) if args.radiomics_path else None,
        clinical_data_path=Path(args.clinical_data),
        gene_gene_path=Path(args.gene_gene_path),
        gene_structure_path=Path(args.gene_structure_path),
        gene_embeddings_path=Path(args.gene_embeddings_dir),
        bold_threshold=config['bold_threshold'],
        verbose=args.verbose
    )
    
    # Step 3: Load clinical data
    print(f"Loading clinical data from {args.clinical_data}")
    clinical_df = pd.read_csv(args.clinical_data)
    
    # Step 4: Determine patients to process
    if args.patient_ids:
        patient_ids = args.patient_ids.split(',')
        print(f"Processing specified {len(patient_ids)} patients")
    else:
        # Get top N patients
        patient_ids = clinical_df['BID'].unique().tolist()[:args.top_n]
        print(f"Processing top {len(patient_ids)} patients")
    
    # Step 5: Prepare patient data
    # Extract clinical features for model input
    patient_features = []
    valid_patient_ids = []
    
    for patient_id in patient_ids:
        try:
            # Get patient features
            patient_row = clinical_df[clinical_df['BID'] == patient_id]
            if len(patient_row) == 0:
                print(f"No clinical data for patient {patient_id}")
                continue
                
            # Get latest visit
            latest_visit = patient_row['VISITCD'].max()
            latest_row = patient_row[patient_row['VISITCD'] == latest_visit].iloc[0]
            
            # Extract numeric features
            numeric_cols = patient_row.select_dtypes(include=[np.number]).columns
            features = np.array(latest_row[numeric_cols], dtype=np.float32)
            
            patient_features.append(features)
            valid_patient_ids.append(patient_id)
            
        except Exception as e:
            print(f"Error preparing data for patient {patient_id}: {str(e)}")
    
    if not valid_patient_ids:
        print("No valid patients found")
        return
    
    # Convert features to tensor
    patient_features_tensor = torch.tensor(np.array(patient_features), dtype=torch.float32).to(device)
    
    # Step 6: Process each patient
    results = {}
    for i, patient_id in enumerate(tqdm(valid_patient_ids, desc="Processing patients")):
        try:
            result = process_patient(
                patient_id,
                graph_builder,
                model,
                patient_features_tensor[i:i+1],
                device
            )
            
            if result:
                results[patient_id] = result
                if args.verbose:
                    print(f"\nPatient {patient_id}:")
                    print(f"  PACC prediction: {result['paccv6_prediction']:.2f}")
                    print(f"  Risk score: {result['risk_score']:.4f}")
            
        except Exception as e:
            print(f"Error in workflow for patient {patient_id}: {str(e)}")
    
    # Step 7: Save results
    if results:
        # Save detailed results
        output_file = output_dir / "prediction_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary CSV
        summary_data = []
        for patient_id, result in results.items():
            summary_data.append({
                'patient_id': patient_id,
                'pacc_prediction': result['paccv6_prediction'],
                'risk_score': result['risk_score'],
                'num_visits': result.get('num_visits', 1)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "prediction_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nResults saved to {output_dir}")
        print(f"  - Detailed results: {output_file}")
        print(f"  - Summary CSV: {summary_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Processed {len(results)}/{len(valid_patient_ids)} patients successfully")
        print(f"Average PACC prediction: {summary_df['pacc_prediction'].mean():.2f}")
        print(f"Min PACC prediction: {summary_df['pacc_prediction'].min():.2f}")
        print(f"Max PACC prediction: {summary_df['pacc_prediction'].max():.2f}")
    else:
        print("No valid results generated")
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BrainTemporalGNN Complete Workflow')
    
    # Model input
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    
    # Data inputs for graph building
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
    parser.add_argument('--output_dir', type=str, default='./workflow_results',
                        help='Directory to save results')
    
    # Processing options
    parser.add_argument('--patient_ids', type=str, default=None,
                        help='Comma-separated list of patient IDs (optional)')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of patients to process if no IDs provided')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    run_workflow(args)