import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import traceback
from tqdm import tqdm

# Import model and utilities
from model import BrainTemporalGNN
from utils import PatientDataset, load_csv_data, setup_patient_graphs


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


def predict_patient(model, patient_graph, patient_data, patient_id, device):
    """
    Make predictions for a single patient.
    
    Args:
        model: The trained model
        patient_graph: Patient's graph data
        patient_data: Patient's clinical data
        patient_id: Patient ID
        device: Computation device
        
    Returns:
        dict: Prediction results with survival and PACC
    """
    model.eval()
    
    try:
        # Move data to device
        patient_graph = patient_graph.to(device)
        
        # Get patient data index
        if hasattr(patient_data, 'patient_ids'):
            try:
                patient_idx = list(patient_data.patient_ids).index(patient_id)
                patient_features = patient_data[patient_idx:patient_idx+1]
            except ValueError:
                print(f"Patient ID {patient_id} not found in dataset")
                return None
        else:
            patient_features = patient_data
            
        # Get predictions
        with torch.no_grad():
            prediction = model.get_prediction_for_patient(
                patient_id, 
                patient_graph, 
                patient_features
            )
            
        if prediction is None:
            return None
            
        # Calculate survival probabilities from DeepHit predictions
        deephit_preds = prediction['deephit_predictions']
        surv_probs = 1 - np.cumsum(deephit_preds)
        
        # Calculate risk score (negative of survival probability at median time)
        median_idx = len(surv_probs) // 2
        risk_score = -surv_probs[median_idx]
        
        # Return simplified results
        return {
            'patient_id': patient_id,
            'paccv6_prediction': float(prediction['paccv6_prediction']),
            'risk_score': float(risk_score),
            'survival_probabilities': surv_probs.tolist()
        }
        
    except Exception as e:
        print(f"Error predicting for patient {patient_id}: {str(e)}")
        traceback.print_exc()
        return None


def run_inference(model_path, data_dir, csv_file, output_file, patient_ids=None, bold_threshold=50):
    """
    Run inference on specified patients.
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing graph data
        csv_file: Path to patient CSV file
        output_file: Path to save results
        patient_ids: List of patient IDs (if None, use all from CSV)
        bold_threshold: BOLD correlation threshold
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model, config = load_model(model_path, device)
    
    # Override bold_threshold if provided
    config['bold_threshold'] = bold_threshold
    
    # Get patient IDs if not provided
    if patient_ids is None:
        try:
            csv_data = pd.read_csv(csv_file)
            patient_ids = csv_data['BID'].unique().tolist()
        except Exception as e:
            print(f"Error loading patient IDs from CSV: {str(e)}")
            return
    
    print(f"Running inference on {len(patient_ids)} patients")
    
    # Load CSV data
    patient_data = load_csv_data(patient_ids, csv_file)
    patient_dataset = PatientDataset(patient_data, patient_ids, device)
    
    # Load graph data
    patient_graphs = setup_patient_graphs(
        patient_ids, 
        data_dir, 
        config['bold_threshold'],
        config['use_gene_edges'],
        config['use_gene_nodes']
    )
    
    print(f"Loaded graphs for {len(patient_graphs)} patients")
    
    # Make predictions
    results = {}
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        if patient_id in patient_graphs:
            prediction = predict_patient(
                model, 
                patient_graphs[patient_id], 
                patient_dataset, 
                patient_id, 
                device
            )
            if prediction:
                results[patient_id] = prediction
    
    # Save results
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save a simple CSV with key results
    df_results = []
    for patient_id, result in results.items():
        df_results.append({
            'patient_id': patient_id,
            'pacc_prediction': result['paccv6_prediction'],
            'risk_score': result['risk_score']
        })
    
    results_df = pd.DataFrame(df_results)
    csv_output = output_file.replace('.pkl', '.csv')
    results_df.to_csv(csv_output, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - Detailed results: {output_file}")
    print(f"  - CSV summary: {csv_output}")
    print(f"Successfully processed {len(results)} patients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple BrainTemporalGNN Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing brain graph data')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to patient CSV file')
    parser.add_argument('--output_file', type=str, default='./predictions.pkl',
                        help='Path to save prediction results')
    parser.add_argument('--patient_ids', type=str, default=None,
                        help='Comma-separated list of patient IDs (optional)')
    parser.add_argument('--bold_threshold', type=int, default=50,
                        help='BOLD correlation threshold (0-100)')
    
    args = parser.parse_args()
    
    # Process patient IDs if provided
    patient_id_list = None
    if args.patient_ids:
        patient_id_list = args.patient_ids.split(',')
    
    run_inference(
        args.model_path,
        args.data_dir,
        args.csv_file,
        args.output_file,
        patient_id_list,
        args.bold_threshold
    )