import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple, NamedTuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import traceback


class PatientSurvivalInfo(NamedTuple):
    """Container for patient survival information"""
    time: float  # TRTENDT_DAYS_T0
    event: int   # 1 if event occurred (SUBJCOMPTR), 0 if censored
    paccv6: float


class BrainGraphBuilder:
    def __init__(
        self, 
        embeddings_base_path: Path,
        correlation_path: Path,
        clinical_data_path: Path,
        gene_gene_path: Path,
        gene_structure_path: Path,
        gene_embeddings_path: Path,
        radiomics_path: Optional[Path] = None,
        bold_threshold: float = 50.0,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.bold_threshold = bold_threshold
        
        # Brain structure names remain the same
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

        # Create node index mapping
        self.node_mapping = {structure: idx for idx, structure in enumerate(self.structures)}
        
        # Define paths
        self.embeddings_base_path = embeddings_base_path
        self.correlation_path = correlation_path
        self.radiomics_path = radiomics_path
        self.clinical_data_path = clinical_data_path
        self.gene_gene_path = gene_gene_path
        self.gene_structure_path = gene_structure_path
        self.gene_embeddings_path = gene_embeddings_path
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load all data
        self.load_correlation_data()
        self.load_clinical_data()
        self.load_gene_data()
        if self.radiomics_path:
            self.load_radiomics_data()

    def load_correlation_data(self):
        """Load precomputed BOLD correlations"""
        try:
            self.correlation_df = pd.read_csv(self.correlation_path)
            if self.verbose:
                print(f"\nCorrelation Data Summary:")
                print(f"Total correlations: {len(self.correlation_df)}")
                print(f"Unique patients: {self.correlation_df['patient_id'].nunique()}")
                print(f"Unique visits: {self.correlation_df['visit_code'].nunique()}")
        except Exception as error:
            print(f"Error loading correlation data: {str(error)}")
            self.correlation_df = None

    def load_clinical_data(self):
        """Load clinical data with survival information"""
        try:
            self.clinical_df = pd.read_csv(self.clinical_data_path)
            if self.verbose:
                print(f"\nClinical Data Summary:")
                print(f"Total rows: {len(self.clinical_df)}")
                print(f"Total unique patients: {len(self.clinical_df['BID'].unique())}")
                print(f"Visit codes: {sorted(self.clinical_df['VISITCD'].unique())}")
        except Exception as error:
            print(f"Error loading clinical data: {str(error)}")
            self.clinical_df = None

    def load_radiomics_data(self):
        try:
            self.radiomics_df = pd.read_csv(self.radiomics_path)
            
            # Create new normalized columns instead of in-place modification
            feature_cols = self.radiomics_df.columns[:107]
            features = self.radiomics_df[feature_cols].astype(float)
            self.normalized_features = (features - features.mean()) / (features.std() + 1e-8)
            
            if self.verbose:
                print(f"\nRadiomics Data Summary:")
                print(f"Total rows: {len(self.radiomics_df)}")
                print(f"Features: {list(self.radiomics_df.columns)}")
                
        except Exception as error:
            print(f"Error loading radiomics data: {str(error)}")
            self.radiomics_df = None

    def get_patient_survival_info(self, patient_id: str, visit_code: Union[str, int]) -> Optional[PatientSurvivalInfo]:
        """Get survival information for a specific patient and visit"""
        if self.clinical_df is None:
            return None
            
        try:
            visit_int = int(visit_code)
            mask = (self.clinical_df['BID'] == patient_id) & (self.clinical_df['VISITCD'] == visit_int)
            if not mask.any():
                return None
                
            row = self.clinical_df.loc[mask].iloc[0]
            return PatientSurvivalInfo(
                time=float(row['TRTENDT_DAYS_T0']),
                event=int(row['SUBJCOMPTR'] == 1),
                paccv6=float(row['PACCV6'])
            )
        except Exception as error:
            if self.verbose:
                print(f"Error getting survival info: {str(error)}")
            return None

    def get_radiomics_features(self, patient_id: str, visit_code: Union[str, int], structure: str) -> Optional[torch.Tensor]:
        if self.radiomics_df is None:
            return None
            
        try:
            formatted_visit = self.format_visit_code(visit_code)
            key = f"{patient_id}_{formatted_visit}"
            struct = structure.lower().replace("_", " ")
            mask = ((self.radiomics_df['Key'] == key) & 
                (self.radiomics_df['Structure'] == struct))
            
            if not mask.any():
                if self.verbose:
                    print(f"No radiomics data found for {key}, structure {structure}")
                return None
                
            # Use normalized features instead
            row_idx = mask.idxmax()
            features = self.normalized_features.iloc[row_idx].values
            return torch.from_numpy(features).unsqueeze(0)
            
        except Exception as error:
            if self.verbose:
                print(f"Error getting radiomics features: {str(error)}")
            return None

    def format_visit_code(self, visit_code: Union[str, int]) -> str:
        """Convert visit code to proper format"""
        visit_int = int(visit_code)
        return f"{visit_int:03d}"

    def load_structure_embeddings(self, patient_id: str, visit_code: Union[str, int]) -> Dict[str, torch.Tensor]:
        """Load structure embeddings with enhanced error handling and debug information"""
        try:
            embeddings = {}
            formatted_visit = self.format_visit_code(visit_code)
            patient_folder = f"A4_MR_T1_{patient_id}_{formatted_visit}"
            emb_path = self.embeddings_base_path / patient_folder
            
            if not emb_path.exists():
                if self.verbose:
                    print(f"\nEmbedding path does not exist: {emb_path}")
                    print(f"Base path exists: {self.embeddings_base_path.exists()}")
                    if self.embeddings_base_path.exists():
                        print("Available patient folders:")
                        for folder in sorted(self.embeddings_base_path.iterdir())[:5]:
                            print(f"  - {folder.name}")
                raise FileNotFoundError(f"No embeddings folder found for {patient_folder}")
            
            if self.verbose:
                print(f"\nLoading embeddings from: {emb_path}")
                
            for structure in self.structures:
                pth_file = emb_path / f"A4_MR_T1_{patient_id}_{formatted_visit}_{structure}.pth"
                if pth_file.exists():
                    try:
                        # Load embedding
                        embedding = torch.load(pth_file, weights_only=True, map_location=self.device)
                        
                        # Standardize dimensions - ensure it's 2D [1, 512]
                        if embedding.dim() == 1:
                            embedding = embedding.unsqueeze(0)  # Add batch dimension
                        elif embedding.dim() > 2:
                            # If more than 2D, flatten to [1, -1] then project if needed
                            embedding = embedding.view(1, -1)
                            if embedding.size(1) != 512:
                                projection = nn.Linear(embedding.size(1), 512).to(self.device)
                                with torch.no_grad():
                                    embedding = projection(embedding)
                        
                        # Verify final shape
                        if embedding.shape[1] != 512:
                            if self.verbose:
                                print(f"Invalid embedding shape for {structure}: {embedding.shape}")
                            continue
                            
                        embeddings[structure] = embedding
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading embedding for {structure}: {str(e)}")
                elif self.verbose:
                    print(f"Missing embedding file: {pth_file.name}")
                        
            if not embeddings:
                structures_found = list(embeddings.keys())
                if self.verbose:
                    print(f"\nNo valid embeddings found in {emb_path}")
                    print(f"Structures found: {structures_found}")
                    print("\nAvailable files:")
                    for item in emb_path.iterdir():
                        print(f"  - {item.name}")
                raise ValueError(f"No valid embeddings found in {patient_folder}")
                
            if self.verbose:
                print(f"\nSuccessfully loaded {len(embeddings)} embeddings")
                print(f"Structures with embeddings: {sorted(embeddings.keys())}")
                
            return embeddings
            
        except Exception as error:
            if self.verbose:
                print(f"\nError in load_structure_embeddings: {str(error)}")
                print(f"Patient ID: {patient_id}")
                print(f"Visit code: {visit_code}")
            return None

    def load_gene_data(self):
        """Load gene-gene and gene-structure connections"""
        try:
            # Load gene-gene connections
            self.gene_gene_df = pd.read_csv(self.gene_gene_path)
            
            # Load gene-structure connections
            self.gene_structure_df = pd.read_csv(self.gene_structure_path)
            
            # Get unique genes
            genes1 = set(self.gene_gene_df['Gene 1'].unique())
            genes2 = set(self.gene_gene_df['Gene 2'].unique())
            struct_genes = set(self.gene_structure_df['Gene'].unique())
            self.unique_genes = sorted(list(genes1.union(genes2, struct_genes)))
            
            # Create gene index mapping
            self.gene_mapping = {gene: idx for idx, gene in enumerate(self.unique_genes)}
            
            if self.verbose:
                print(f"\nGene Data Summary:")
                print(f"Total gene-gene connections: {len(self.gene_gene_df)}")
                print(f"Total gene-structure connections: {len(self.gene_structure_df)}")
                print(f"Total unique genes: {len(self.unique_genes)}")
                
        except Exception as error:
            print(f"Error loading gene data: {str(error)}")
            self.gene_gene_df = None
            self.gene_structure_df = None
            self.unique_genes = []
            self.gene_mapping = {}

    def load_gene_embeddings(self, patient_id: str) -> Dict[str, torch.Tensor]:
        """Load gene embeddings for a specific patient and project to 512 dimensions"""
        embeddings = {}
        patient_gene_path = self.gene_embeddings_path / patient_id
        
        try:
            # Create projection layer if it doesn't exist
            if not hasattr(self, 'gene_projection'):
                self.gene_projection = nn.Linear(768, 512).to(self.device)
                # Initialize the projection layer
                nn.init.orthogonal_(self.gene_projection.weight)
                nn.init.zeros_(self.gene_projection.bias)
            
            for gene in self.unique_genes:
                emb_file = patient_gene_path / f"{patient_id}_{gene}_embedding.pt"
                if emb_file.exists():
                    try:
                        # Load embedding with weights_only=True for security
                        embedding = torch.load(emb_file, map_location=self.device, weights_only=True)
                        
                        # Ensure embedding is 2D with shape [1, 768]
                        if embedding.dim() == 1:
                            embedding = embedding.unsqueeze(0)
                        elif embedding.dim() > 2:
                            embedding = embedding.view(1, -1)
                        
                        # Project from 768 to 512 dimensions
                        if embedding.size(1) == 768:
                            with torch.no_grad():
                                embedding = self.gene_projection(embedding)
                        elif embedding.size(1) != 512:
                            if self.verbose:
                                print(f"Warning: Unexpected embedding size for gene {gene}: {embedding.shape}")
                            continue
                        
                        embeddings[gene] = embedding
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading embedding for gene {gene}: {str(e)}")
                        continue
                else:
                    if self.verbose:
                        print(f"Missing embedding for gene {gene}")
            
            if not embeddings:
                raise ValueError(f"No valid gene embeddings found for patient {patient_id}")
                
        except Exception as error:
            if self.verbose:
                print(f"Error loading gene embeddings: {str(error)}")
            return None
        
        return embeddings
    
    def get_visit_correlations(self, patient_id: str, visit_code: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get precomputed correlations for a specific visit"""
        visit_int = int(visit_code)
        
        # Get correlations for this visit
        mask = ((self.correlation_df['patient_id'] == patient_id) & 
                (self.correlation_df['visit_code'] == visit_int))
        visit_corrs = self.correlation_df[mask]
        
        edge_indices = []
        edge_weights = []
        
        # Calculate threshold for this visit
        threshold = np.percentile(visit_corrs['correlation'], self.bold_threshold)
        if self.bold_threshold == 100:
            threshold = -np.inf
        else:
            threshold = np.percentile(visit_corrs['correlation'], 100 - self.bold_threshold)
        
        for _, row in visit_corrs.iterrows():
            correlation = row['correlation']
            
            # Only add edges above threshold
            if abs(correlation) >= threshold:
                struct1 = row['structure1']
                struct2 = row['structure2']
                idx1 = self.node_mapping[struct1]
                idx2 = self.node_mapping[struct2]
                
                # Add bidirectional edges
                edge_indices.extend([[idx1, idx2], [idx2, idx1]])
                edge_weights.extend([correlation, correlation])
        
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        return (
            torch.tensor(edge_indices, dtype=torch.long).t(),
            torch.tensor(edge_weights, dtype=torch.float)
        )

    def get_stage_genes(self, stage: str) -> set:
        """Get set of genes present in a specific stage"""
        if self.gene_structure_df is None:
            return set()
        return set(self.gene_structure_df[self.gene_structure_df['Stage'] == stage]['Gene'].unique())

    def get_gene_edges(self, stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get gene-gene connections for genes present in the same stage"""
        if self.gene_gene_df is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        # Get genes present in this stage
        stage_genes = self.get_stage_genes(stage)
        
        edge_indices = []
        edge_weights = []
        
        for _, row in self.gene_gene_df.iterrows():
            gene1, gene2 = row['Gene 1'], row['Gene 2']
            
            # Only add connection if both genes are present in this stage
            if gene1 in stage_genes and gene2 in stage_genes:
                gene1_idx = self.gene_mapping[gene1]
                gene2_idx = self.gene_mapping[gene2]
                weight = float(row['Weight'])
                
                # Add bidirectional edges
                edge_indices.extend([[gene1_idx, gene2_idx], [gene2_idx, gene1_idx]])
                edge_weights.extend([weight, weight])
            
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        return (
            torch.tensor(edge_indices, dtype=torch.long).t(),
            torch.tensor(edge_weights, dtype=torch.float)
        )

    def get_gene_structure_edges(self, stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get gene-structure connections for a specific stage"""
        if self.gene_structure_df is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        edge_indices = []
        edge_weights = []
        
        # Filter by stage
        stage_df = self.gene_structure_df[self.gene_structure_df['Stage'] == stage]
        
        for _, row in stage_df.iterrows():
            if row['Gene'] in self.gene_mapping and row['Brain_Structures'] in self.node_mapping:
                gene_idx = self.gene_mapping[row['Gene']]
                struct_idx = self.node_mapping[row['Brain_Structures']]
                weight = float(row['Weight'])
                
                # Add bidirectional edges
                edge_indices.extend([[gene_idx, struct_idx], [struct_idx, gene_idx]])
                edge_weights.extend([weight, weight])
        
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        return (
            torch.tensor(edge_indices, dtype=torch.long).t(),
            torch.tensor(edge_weights, dtype=torch.float)
        )

    def build_patient_graph(self, patient_id: str) -> HeteroData:
        try:
            data = HeteroData()
            patient_visits = (self.clinical_df[self.clinical_df['BID'] == patient_id]
                            ['VISITCD'].sort_values().unique())
            
            if len(patient_visits) == 0:
                raise ValueError(f"No visits found for patient {patient_id}")
            
            # Load gene embeddings for this patient
            gene_embeddings = self.load_gene_embeddings(patient_id)
            if gene_embeddings is None:
                gene_features = torch.zeros((len(self.unique_genes), 512))
            else:
                gene_features = torch.stack([
                    gene_embeddings.get(gene, torch.zeros(1, 512)).squeeze(0)
                    for gene in self.unique_genes
                ])
            
            # Add gene features to the graph
            data['gene'].x = gene_features
            
            # Process structure features and edges
            all_node_features = []
            spatial_edges = []
            temporal_edges = []
            node_offset = 0
            
            for visit_idx, visit in enumerate(patient_visits):
                structure_embeddings = self.load_structure_embeddings(patient_id, visit)
                if structure_embeddings is None:
                    # Skip this visit if no embeddings found
                    continue
                    
                visit_features = []
                
                for structure in self.structures:
                    embedding = structure_embeddings.get(structure, torch.zeros(1, 512)).squeeze(0)
                    visit_features.append(embedding)
                
                # Add spatial edges
                edge_index, edge_weights = self.get_visit_correlations(patient_id, visit)
                edge_index = edge_index + node_offset
                spatial_edges.append((edge_index, edge_weights))
                
                # Add temporal edges
                if visit_idx > 0:
                    prev_visit = patient_visits[visit_idx - 1]
                    temp_edges = []
                    edge_features = []
                    
                    for idx, structure in enumerate(self.structures):
                        curr_radiomics = self.get_radiomics_features(patient_id, visit, structure)
                        prev_radiomics = self.get_radiomics_features(patient_id, prev_visit, structure)
                        
                        if curr_radiomics is not None and prev_radiomics is not None:
                            curr_idx = node_offset + idx
                            prev_idx = node_offset - len(self.structures) + idx
                            temp_edges.append([curr_idx, prev_idx])
                            edge_features.append(curr_radiomics.squeeze(0) - prev_radiomics.squeeze(0))
                        else:
                            # If radiomics not available, use dummy features
                            curr_idx = node_offset + idx
                            prev_idx = node_offset - len(self.structures) + idx
                            temp_edges.append([curr_idx, prev_idx])
                            edge_features.append(torch.zeros(107))
                    
                    if temp_edges:
                        temporal_edges.append((
                            torch.tensor(temp_edges).t(),
                            torch.stack(edge_features)
                        ))
                
                num_visits = len(patient_visits)

                if num_visits == 1:
                    # For single visit, add all stages (Early, Mid, and Late)
                    stages = ["Early", "Mid", "Late"]
                    for stage in stages:
                        # Add gene-gene connections for all stages
                        gene_edge_index, gene_edge_weights = self.get_gene_edges(stage)
                        if gene_edge_index.shape[1] > 0:
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_index = gene_edge_index
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_attr = gene_edge_weights.view(-1, 1)
                        
                        # Add gene-structure connections for all stages
                        gene_struct_edge_index, gene_struct_weights = self.get_gene_structure_edges(stage)
                        if gene_struct_edge_index.shape[1] > 0:
                            struct_edges = gene_struct_edge_index.clone()
                            struct_edges[1] += node_offset  # Offset structure indices
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_index = struct_edges
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_attr = gene_struct_weights.view(-1, 1)

                elif num_visits == 2:
                    if visit_idx == 0:
                        # First visit gets Early and Midpoint
                        stages = ["Early", "Mid"]
                    else:
                        # Last visit gets Late
                        stages = ["Late"]
                    
                    for stage in stages:
                        # Add gene-gene connections
                        gene_edge_index, gene_edge_weights = self.get_gene_edges(stage)
                        if gene_edge_index.shape[1] > 0:
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_index = gene_edge_index
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_attr = gene_edge_weights.view(-1, 1)
                        
                        # Add gene-structure connections
                        gene_struct_edge_index, gene_struct_weights = self.get_gene_structure_edges(stage)
                        if gene_struct_edge_index.shape[1] > 0:
                            struct_edges = gene_struct_edge_index.clone()
                            struct_edges[1] += node_offset  # Offset structure indices
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_index = struct_edges
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_attr = gene_struct_weights.view(-1, 1)

                else:
                    # Original logic for 3 or more visits
                    if visit_idx == 0:
                        stage = "Early"
                    elif visit_idx == len(patient_visits) - 1:
                        stage = "Late"
                    elif visit_idx == len(patient_visits) // 2:
                        stage = "Mid"
                    else:
                        stage = None
                    
                    if stage is not None:
                        # Add gene-gene connections for this stage
                        gene_edge_index, gene_edge_weights = self.get_gene_edges(stage)
                        if gene_edge_index.shape[1] > 0:
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_index = gene_edge_index
                            data['gene', f'interacts_{stage.lower()}', 'gene'].edge_attr = gene_edge_weights.view(-1, 1)
                        
                        # Add gene-structure connections for this stage
                        gene_struct_edge_index, gene_struct_weights = self.get_gene_structure_edges(stage)
                        if gene_struct_edge_index.shape[1] > 0:
                            struct_edges = gene_struct_edge_index.clone()
                            struct_edges[1] += node_offset  # Offset structure indices
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_index = struct_edges
                            data['gene', f'connects_{stage.lower()}', 'structure'].edge_attr = gene_struct_weights.view(-1, 1)
                
                all_node_features.extend(visit_features)
                node_offset += len(self.structures)
            
            # Add structure features and edges to the graph
            data['structure'].x = torch.stack(all_node_features)
            
            if spatial_edges:
                all_spatial_indices = torch.cat([e[0] for e in spatial_edges], dim=1)
                all_spatial_weights = torch.cat([e[1] for e in spatial_edges])
                data['structure', 'bold_correlated', 'structure'].edge_index = all_spatial_indices
                data['structure', 'bold_correlated', 'structure'].edge_attr = all_spatial_weights.view(-1, 1)
            
            if temporal_edges:
                all_temp_indices = torch.cat([e[0] for e in temporal_edges], dim=1)
                all_temp_features = torch.cat([e[1] for e in temporal_edges])
                data['structure', 'temporally_connected', 'structure'].edge_index = all_temp_indices
                data['structure', 'temporally_connected', 'structure'].edge_attr = all_temp_features
            
            # Add metadata
            data.patient_id = patient_id
            data.visits = patient_visits.tolist()
            
            # Add survival information
            times = []
            events = []
            paccv6_scores = []
            for visit in patient_visits:
                survival_info = self.get_patient_survival_info(patient_id, visit)
                if survival_info:
                    times.append(survival_info.time)
                    events.append(survival_info.event)
                    paccv6_scores.append(survival_info.paccv6)
                    
            data.survival_times = torch.tensor(times)
            data.events = torch.tensor(events)
            data.paccv6_scores = torch.tensor(paccv6_scores)
            
            return data
            
        except Exception as error:
            if self.verbose:
                print(f"Error in build_patient_graph: {str(error)}")
                traceback.print_exc()
            raise