import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import traceback
from tqdm import tqdm

from model import BrainTemporalGNN
from utils import collate_hetero_data


def load_model(model_path, device=None):
    """
    Load a trained BrainTemporalGNN model.
    
    Args:
        model_path (str): Path to the saved model file
        device (torch.device): Device to load the model on
        
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


def predict_single_patient(model, patient_graph, patient_dataset, patient_id, device=None):
    """
    Make predictions for a single patient.
    
    Args:
        model (BrainTemporalGNN): The trained model
        patient_graph (HeteroData): The patient's graph data
        patient_dataset (PatientDataset): Dataset containing patient features
        patient_id (str): Patient ID
        device (torch.device): Device for computation
        
    Returns:
        dict: Prediction results for the patient
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    
    try:
        # Move data to device
        patient_graph = patient_graph.to(device)
        
        # Get patient data
        patient_idx = list(patient_dataset.patient_ids).index(patient_id)
        patient_data = patient_dataset[patient_idx:patient_idx+1]
        
        with torch.no_grad():
            # Use the model's patient-specific prediction method
            prediction = model.get_prediction_for_patient(
                patient_id, 
                patient_graph, 
                patient_data
            )
            
            if prediction is None:
                print(f"Warning: No prediction generated for patient {patient_id}")
                return None
                
            # Calculate survival probabilities from DeepHit predictions
            deephit_preds = prediction['deephit_predictions']
            surv_probs = 1 - np.cumsum(deephit_preds)
            
            # Package results
            result = {
                'paccv6_prediction': prediction['paccv6_prediction'],
                'deephit_predictions': deephit_preds,
                'survival_probabilities': surv_probs,
                'shared_features': prediction['shared_features'],
                'num_visits': len(patient_graph.visits) if hasattr(patient_graph, 'visits') else 1,
                'risk_score': -surv_probs[len(surv_probs)//2]  # Use middle point as risk score
            }
            
            return result
            
    except Exception as e:
        print(f"Error predicting for patient {patient_id}: {str(e)}")
        traceback.print_exc()
        return None


def predict_batch(model, patient_graphs, patient_dataset, batch_size=4, device=None):
    """
    Make predictions for a batch of patients.
    
    Args:
        model (BrainTemporalGNN): The trained model
        patient_graphs (dict): Dictionary of patient graphs keyed by patient ID
        patient_dataset (PatientDataset): Dataset containing patient features
        batch_size (int): Batch size for inference
        device (torch.device): Device for computation
        
    Returns:
        dict: Prediction results for all patients
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    results = {}
    
    try:
        # Prepare batches of patient IDs
        patient_ids = list(patient_graphs.keys())
        num_batches = (len(patient_ids) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patient_ids))
            batch_patient_ids = patient_ids[batch_start:batch_end]
            
            # Get batch graphs
            batch_graphs = [patient_graphs[pid] for pid in batch_patient_ids if pid in patient_graphs]
            
            if not batch_graphs:
                continue
                
            # Create graph dataloader
            graph_loader = DataLoader(
                batch_graphs, 
                batch_size=len(batch_graphs), 
                shuffle=False, 
                collate_fn=collate_hetero_data
            )
            
            # Get patient indices for the batch
            patient_indices = []
            for pid in batch_patient_ids:
                try:
                    idx = list(patient_dataset.patient_ids).index(pid)
                    patient_indices.append(idx)
                except ValueError:
                    print(f"Warning: Patient {pid} not found in patient dataset")
                    
            if not patient_indices:
                continue
                
            # Get patient data for the batch
            batch_patient_data = torch.stack([patient_dataset[idx] for idx in patient_indices])
            
            with torch.no_grad():
                # Forward pass
                for graph_batch in graph_loader:
                    graph_batch = graph_batch.to(device)
                    
                    shared_features, deephit_preds, paccv6_preds = model(
                        graph_batch, 
                        batch_patient_data.to(device)
                    )
                    
                    if shared_features is None:
                        continue
                        
                    # Process results for each patient in the batch
                    for i, pid in enumerate(batch_patient_ids):
                        if i >= len(shared_features):
                            continue
                            
                        # Calculate survival probabilities
                        deephit_pred = deephit_preds[i].cpu().numpy()
                        surv_probs = 1 - np.cumsum(deephit_pred)
                        
                        # Package results
                        results[pid] = {
                            'paccv6_prediction': paccv6_preds[i].item(),
                            'deephit_predictions': deephit_pred,
                            'survival_probabilities': surv_probs,
                            'shared_features': shared_features[i].cpu().numpy(),
                            'num_visits': len(patient_graphs[pid].visits) if hasattr(patient_graphs[pid], 'visits') else 1,
                            'risk_score': -surv_probs[len(surv_probs)//2]  # Use middle point as risk score
                        }
        
        return results
        
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        traceback.print_exc()
        return results


def calculate_feature_importance(model, patient_graphs, patient_dataset, device=None):
    """
    Calculate feature importance for the model using permutation importance.
    
    Args:
        model (BrainTemporalGNN): The trained model
        patient_graphs (dict): Dictionary of patient graphs keyed by patient ID
        patient_dataset (PatientDataset): Dataset containing patient features
        device (torch.device): Device for computation
        
    Returns:
        dict: Feature importance scores
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    
    try:
        # Prepare graph data for importance calculation
        patient_ids = list(patient_graphs.keys())
        graphs = [patient_graphs[pid] for pid in patient_ids if pid in patient_graphs]
        
        if not graphs:
            return None
            
        # Create dataloader
        graph_loader = DataLoader(
            graphs, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate_hetero_data
        )
        
        # Calculate importance using model's built-in method
        importance_scores = {}
        
        for patient_id, graph in zip(patient_ids, graphs):
            # Get patient data
            try:
                patient_idx = list(patient_dataset.patient_ids).index(patient_id)
                patient_data = patient_dataset[patient_idx:patient_idx+1]
                
                # Calculate feature importance
                importance = model.get_feature_importance(
                    graph.to(device), 
                    patient_data.to(device)
                )
                
                if importance:
                    importance_scores[patient_id] = importance
                    
            except Exception as e:
                print(f"Error calculating importance for patient {patient_id}: {str(e)}")
                continue
                
        return importance_scores
        
    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        traceback.print_exc()
        return None