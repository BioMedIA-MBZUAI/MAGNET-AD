"""
MAGNET-AD Inference Script
Provides inference capabilities for trained MAGNET models.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from model import MAGNET
from utils import load_csv_data, PatientDataset, collate_hetero_data
from graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAGNETInference:
    """
    MAGNET-AD Inference Engine
    
    Provides capabilities for:
    1. Loading trained models
    2. Making predictions on new patients
    3. Feature importance analysis
    4. Attention visualization
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info("MAGNET inference engine initialized")

    def _load_model(self) -> MAGNET:
        """Load the trained MAGNET model"""
        try:
            # Initialize model with config parameters
            model = MAGNET(
                structure_input_dim=self.config['model']['structure_input_dim'],
                gene_input_dim=self.config['model']['gene_input_dim'],
                hidden_channels=self.config['model']['hidden_channels'],
                num_spatial_layers=self.config['model']['num_spatial_layers'],
                num_heads=self.config['model']['num_heads'],
                dropout=0.0,  # No dropout during inference
                csv_input_dim=100,  # Will be adjusted based on actual data
                csv_hidden_dim=self.config['model']['csv_hidden_dim'],
                csv_output_dim=self.config['model']['csv_output_dim'],
                deephit_duration_index=self.config['model']['deephit_duration_index'],
                use_temporal_dynamics=self.config['model']['use_temporal_dynamics']
            ).to(self.device)
            
            # Load trained weights
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict_patient(
        self, 
        patient_graph, 
        patient_clinical_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for a single patient.
        
        Args:
            patient_graph: HeteroData graph for the patient
            patient_clinical_data: Clinical data array
            
        Returns:
            Dictionary containing predictions and features
        """
        try:
            with torch.no_grad():
                # Move data to device
                patient_graph = patient_graph.to(self.device)
                clinical_tensor = torch.from_numpy(patient_clinical_data).float().to(self.device)
                
                # Add batch dimension
                clinical_tensor = clinical_tensor.unsqueeze(0)
                
                # Forward pass
                shared_features, deephit_preds, pacc_preds, sequence_preds, temporal_weights = self.model(
                    patient_graph, clinical_tensor
                )
                
                if shared_features is None:
                    raise ValueError("Model forward pass failed")
                
                # Process predictions
                results = {}
                
                # Survival predictions (probability mass function)
                if deephit_preds is not None:
                    pmf = torch.softmax(deephit_preds, dim=1).cpu().numpy()
                    survival_probs = 1 - np.cumsum(pmf, axis=1)
                    
                    results['survival_pmf'] = pmf.squeeze()
                    results['survival_function'] = survival_probs.squeeze()
                    results['predicted_risk'] = np.mean(survival_probs.squeeze())
                
                # PACC predictions
                if pacc_preds is not None:
                    results['pacc_prediction'] = pacc_preds.cpu().numpy().squeeze()
                
                # Sequence predictions for temporal analysis
                if sequence_preds:
                    sequence_arrays = []
                    for seq in sequence_preds:
                        if isinstance(seq, list):
                            seq_array = [s.cpu().numpy() if torch.is_tensor(s) else s for s in seq]
                            sequence_arrays.append(seq_array)
                    results['sequence_predictions'] = sequence_arrays
                
                # Shared features for downstream analysis
                results['shared_features'] = shared_features.cpu().numpy().squeeze()
                
                # Temporal weights
                if temporal_weights is not None:
                    results['temporal_weights'] = temporal_weights.cpu().numpy().squeeze()
                
                return results
                
        except Exception as e:
            logger.error(f"Error in patient prediction: {e}")
            raise

    def predict_batch(
        self,
        patient_graphs: List,
        clinical_data: np.ndarray,
        batch_size: int = 8
    ) -> List[Dict[str, np.ndarray]]:
        """
        Make predictions for multiple patients in batches.
        
        Args:
            patient_graphs: List of patient graphs
            clinical_data: Clinical data array [num_patients, num_features]
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(patient_graphs), batch_size):
            batch_graphs = patient_graphs[i:i+batch_size]
            batch_clinical = clinical_data[i:i+batch_size]
            
            # Process each patient in the batch
            for j, (graph, clinical) in enumerate(zip(batch_graphs, batch_clinical)):
                try:
                    patient_results = self.predict_patient(graph, clinical)
                    patient_results['patient_index'] = i + j
                    results.append(patient_results)
                except Exception as e:
                    logger.warning(f"Failed to process patient {i+j}: {e}")
                    continue
        
        return results

    def compute_feature_importance(
        self,
        patient_graph,
        patient_clinical_data: np.ndarray,
        method: str = 'perturbation'
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance scores for model interpretability.
        
        Args:
            patient_graph: Patient graph data
            patient_clinical_data: Clinical data
            method: Importance computation method ('perturbation', 'gradient')
            
        Returns:
            Feature importance scores
        """
        try:
            if method == 'perturbation':
                return self._compute_perturbation_importance(patient_graph, patient_clinical_data)
            elif method == 'gradient':
                return self._compute_gradient_importance(patient_graph, patient_clinical_data)
            else:
                raise ValueError(f"Unknown importance method: {method}")
                
        except Exception as e:
            logger.error(f"Error computing feature importance: {e}")
            return {}

    def _compute_perturbation_importance(
        self,
        patient_graph,
        patient_clinical_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute feature importance using perturbation analysis"""
        
        # Get baseline prediction
        baseline_results = self.predict_patient(patient_graph, patient_clinical_data)
        baseline_pacc = baseline_results.get('pacc_prediction', 0.0)
        
        importance_scores = {}
        
        # Brain region importance (perturb each of 32 regions)
        if 'structure' in patient_graph.node_types:
            structure_importance = []
            
            for region_idx in range(32):  # 32 brain regions
                # Create perturbed graph
                perturbed_graph = patient_graph.clone()
                
                # Zero out features for this region across all visits
                num_structures = perturbed_graph['structure'].x.size(0)
                region_mask = torch.arange(num_structures) % 32 == region_idx
                perturbed_graph['structure'].x[region_mask] = 0
                
                # Get prediction with perturbation
                try:
                    perturbed_results = self.predict_patient(perturbed_graph, patient_clinical_data)
                    perturbed_pacc = perturbed_results.get('pacc_prediction', 0.0)
                    
                    # Importance = change in prediction
                    importance = abs(baseline_pacc - perturbed_pacc)
                    structure_importance.append(importance)
                    
                except:
                    structure_importance.append(0.0)
            
            importance_scores['brain_regions'] = np.array(structure_importance)
        
        # Gene importance (if gene nodes present)
        if 'gene' in patient_graph.node_types and patient_graph['gene'].x.size(0) > 0:
            gene_importance = []
            
            for gene_idx in range(patient_graph['gene'].x.size(0)):
                # Create perturbed graph
                perturbed_graph = patient_graph.clone()
                perturbed_graph['gene'].x[gene_idx] = 0
                
                # Get prediction with perturbation
                try:
                    perturbed_results = self.predict_patient(perturbed_graph, patient_clinical_data)
                    perturbed_pacc = perturbed_results.get('pacc_prediction', 0.0)
                    
                    importance = abs(baseline_pacc - perturbed_pacc)
                    gene_importance.append(importance)
                    
                except:
                    gene_importance.append(0.0)
            
            importance_scores['genes'] = np.array(gene_importance)
        
        # Clinical feature importance
        clinical_importance = []
        for feature_idx in range(len(patient_clinical_data)):
            perturbed_clinical = patient_clinical_data.copy()
            perturbed_clinical[feature_idx] = 0
            
            try:
                perturbed_results = self.predict_patient(patient_graph, perturbed_clinical)
                perturbed_pacc = perturbed_results.get('pacc_prediction', 0.0)
                
                importance = abs(baseline_pacc - perturbed_pacc)
                clinical_importance.append(importance)
                
            except:
                clinical_importance.append(0.0)
        
        importance_scores['clinical_features'] = np.array(clinical_importance)
        
        return importance_scores

    def _compute_gradient_importance(
        self,
        patient_graph,
        patient_clinical_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute feature importance using gradient analysis"""
        self.model.train()  # Enable gradients
        
        try:
            # Move data to device and enable gradients
            patient_graph = patient_graph.to(self.device)
            clinical_tensor = torch.from_numpy(patient_clinical_data).float().to(self.device)
            clinical_tensor = clinical_tensor.unsqueeze(0).requires_grad_(True)
            
            # Forward pass
            shared_features, deephit_preds, pacc_preds, sequence_preds, temporal_weights = self.model(
                patient_graph, clinical_tensor
            )
            
            if pacc_preds is None:
                return {}
            
            # Compute gradients with respect to PACC prediction
            pacc_preds.backward(retain_graph=True)
            
            importance_scores = {}
            
            # Clinical feature gradients
            if clinical_tensor.grad is not None:
                clinical_gradients = torch.abs(clinical_tensor.grad).cpu().numpy().squeeze()
                importance_scores['clinical_features'] = clinical_gradients
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error in gradient importance: {e}")
            return {}
        
        finally:
            self.model.eval()  # Return to eval mode

    def analyze_temporal_patterns(
        self,
        patient_graph,
        patient_clinical_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Analyze temporal patterns in patient progression.
        
        Args:
            patient_graph: Patient graph with temporal connections
            patient_clinical_data: Clinical data
            
        Returns:
            Temporal analysis results
        """
        try:
            # Get predictions including sequence predictions
            results = self.predict_patient(patient_graph, patient_clinical_data)
            
            analysis = {}
            
            # Temporal progression analysis
            if 'sequence_predictions' in results:
                sequences = results['sequence_predictions']
                
                for seq_idx, seq in enumerate(sequences):
                    if len(seq) >= 2:
                        # Calculate progression rate
                        seq_array = np.array(seq)
                        progression_rate = np.diff(seq_array)
                        
                        analysis[f'progression_rate_seq_{seq_idx}'] = progression_rate
                        analysis[f'total_change_seq_{seq_idx}'] = seq_array[-1] - seq_array[0]
                        analysis[f'trend_seq_{seq_idx}'] = 'increasing' if progression_rate[-1] > 0 else 'decreasing'
            
            # Visit-based analysis
            if hasattr(patient_graph, 'visits'):
                num_visits = len(patient_graph.visits)
                analysis['num_visits'] = num_visits
                analysis['visit_span'] = patient_graph.visits[-1] - patient_graph.visits[0] if num_visits > 1 else 0
            
            # Temporal weights analysis
            if 'temporal_weights' in results:
                temporal_weights = results['temporal_weights']
                analysis['temporal_importance'] = np.mean(temporal_weights)
                analysis['temporal_variability'] = np.std(temporal_weights)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return {}

    def save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to file"""
        try:
            output_path = Path(output_path)
            
            if output_path.suffix == '.pkl':
                with open(output_path, 'wb') as f:
                    pickle.dump(predictions, f)
            elif output_path.suffix == '.csv':
                # Convert to DataFrame for CSV export
                flattened_results = []
                for i, pred in enumerate(predictions):
                    row = {'patient_index': i}
                    
                    # Add scalar predictions
                    for key, value in pred.items():
                        if np.isscalar(value):
                            row[key] = value
                        elif isinstance(value, np.ndarray) and value.size == 1:
                            row[key] = value.item()
                    
                    flattened_results.append(row)
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")


def main():
    parser = argparse.ArgumentParser(description='MAGNET-AD Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to training config')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with graph data')
    parser.add_argument('--clinical_csv', type=str, required=True, help='Clinical data CSV')
    parser.add_argument('--patient_ids', type=str, help='Comma-separated patient IDs')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--compute_importance', action='store_true', help='Compute feature importance')
    parser.add_argument('--analyze_temporal', action='store_true', help='Analyze temporal patterns')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = MAGNETInference(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("MAGNET-AD Inference completed")


if __name__ == '__main__':
    main()
