"""
MAGNET-AD: Heterogeneous Graph Construction
Stage 2: Heterogeneous Graph Construction

Implementation following the exact paper description:
- Brain structure nodes (blue): 32 anatomical regions with 107-dimensional radiomic features
- Gene nodes (green): 100 AD-associated genes with 768-dimensional feature embeddings
- Structure-to-structure edges: BOLD signal correlations with Pearson correlations > threshold
- Gene-to-gene edges: co-expression patterns with edge weights reflecting strength
- Gene-to-structure edges: mRNA expression levels with normalized expression levels
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple, NamedTuple, Union
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import traceback
import logging

# Configure logging
logger = logging.getLogger(__name__)


class PatientSurvivalInfo(NamedTuple):
    """Container for patient survival information"""
    time: float  # conversion time
    event: int   # 1 if event occurred, 0 if censored
    pacc: float


class GraphBuilder:
    """
    MAGNET-AD Heterogeneous Graph Builder
    
    Constructs spatial heterogeneous graphs with:
    1. Brain structure nodes (32 anatomical regions) 
    2. Gene nodes (100 AD-associated genes)
    3. Three types of edges as described in the paper
    """
    
    def __init__(
        self, 
        embeddings_base_path: Path,
        correlation_path: Path,
        clinical_data_path: Path,
        gene_gene_path: Path,
        gene_structure_path: Path,
        gene_embeddings_path: Path,
        radiomics_path: Path,
        bold_threshold: float = 50.0,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.bold_threshold = bold_threshold
        
        # 32 anatomical brain regions as described in the paper
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
        self.load_radiomics_data()

    def load_correlation_data(self):
        """Load precomputed BOLD correlations for structure-to-structure edges"""
        try:
            self.correlation_df = pd.read_csv(self.correlation_path)
            if self.verbose:
                print(f"Loaded BOLD correlations: {len(self.correlation_df)} entries")
        except Exception as error:
            print(f"Error loading correlation data: {str(error)}")
            self.correlation_df = None

    def load_clinical_data(self):
        """Load clinical data with survival information"""
        try:
            self.clinical_df = pd.read_csv(self.clinical_data_path)
            if self.verbose:
                print(f"Loaded clinical data: {len(self.clinical_df)} entries")
        except Exception as error:
            print(f"Error loading clinical data: {str(error)}")
            self.clinical_df = None

    def load_radiomics_data(self):
        """Load 107-dimensional radiomic features for brain structure nodes"""
        try:
            self.radiomics_df = pd.read_csv(self.radiomics_path)
            
            # Verify we have 107 feature columns as described in the paper
            feature_cols = [col for col in self.radiomics_df.columns if col not in ['Key', 'Structure']]
            if len(feature_cols) != 107:
                print(f"Warning: Expected 107 radiomic features, found {len(feature_cols)}")
            
            # Store raw features for per-patient normalization (prevents data leakage)
            self.raw_features = self.radiomics_df[feature_cols].astype(float)
            
            if self.verbose:
                print(f"Loaded radiomic features: {len(self.radiomics_df)} entries with {len(feature_cols)} features")
                
        except Exception as error:
            print(f"Error loading radiomics data: {str(error)}")
            # Initialize empty DataFrames to prevent AttributeError
            self.radiomics_df = pd.DataFrame()
            self.raw_features = pd.DataFrame()

    def load_gene_data(self):
        """Load gene-gene co-expression and gene-structure mRNA expression data"""
        try:
            # Load gene-gene co-expression patterns
            self.gene_gene_df = pd.read_csv(self.gene_gene_path)
            
            # Load gene-structure mRNA expression levels
            self.gene_structure_df = pd.read_csv(self.gene_structure_path)
            
            # Get unique genes (should be 100 AD-associated genes as per paper)
            genes1 = set(self.gene_gene_df['Gene 1'].unique())
            genes2 = set(self.gene_gene_df['Gene 2'].unique())
            struct_genes = set(self.gene_structure_df['Gene'].unique())
            self.unique_genes = sorted(list(genes1.union(genes2, struct_genes)))
            
            # Create gene index mapping
            self.gene_mapping = {gene: idx for idx, gene in enumerate(self.unique_genes)}
            
            if self.verbose:
                print(f"Gene Data Summary:")
                print(f"  - Total gene-gene co-expressions: {len(self.gene_gene_df)}")
                print(f"  - Total gene-structure connections: {len(self.gene_structure_df)}")
                print(f"  - Total unique AD-associated genes: {len(self.unique_genes)}")
                
        except Exception as error:
            print(f"Error loading gene data: {str(error)}")
            self.gene_gene_df = None
            self.gene_structure_df = None
            self.unique_genes = []
            self.gene_mapping = {}

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
                time=float(row['TIME']),
                event=int(row['EVENT'] == 1),
                pacc=float(row['PACC'])
            )
        except Exception as error:
            return None

    def get_radiomics_features(self, patient_id: str, visit_code: Union[str, int], structure: str) -> Optional[torch.Tensor]:
        """Get 107-dimensional radiomic features for a specific structure"""
        if self.radiomics_df is None or self.radiomics_df.empty:
            return None
            
        try:
            formatted_visit = self.format_visit_code(visit_code)
            key = f"{patient_id}_{formatted_visit}"
            struct = structure.lower().replace("_", " ")
            mask = ((self.radiomics_df['Key'] == key) & 
                (self.radiomics_df['Structure'] == struct))
            
            if not mask.any():
                return None
                
            # Use raw features and normalize per-patient to prevent data leakage
            row_idx = mask.idxmax()
            features = self.raw_features.iloc[row_idx].values
            
            # Ensure exactly 107 dimensions as described in the paper
            if len(features) != 107:
                # Pad or truncate to 107 dimensions
                if len(features) < 107:
                    features = np.pad(features, (0, 107 - len(features)), 'constant')
                else:
                    features = features[:107]
            
            # Simple normalization to prevent extreme values
            features = (features - features.mean()) / (features.std() + 1e-8)
            
            return torch.from_numpy(features).float()
            
        except Exception as error:
            return None

    def format_visit_code(self, visit_code: Union[str, int]) -> str:
        """Convert visit code to proper format"""
        visit_int = int(visit_code)
        return f"{visit_int:03d}"

    def load_gene_embeddings(self, patient_id: str) -> Dict[str, torch.Tensor]:
        """Load 768-dimensional gene embeddings and project to 512 dimensions"""
        embeddings = {}
        patient_gene_path = self.gene_embeddings_path / patient_id
        
        try:
            # Create projection layer for gene embeddings (768 -> 512)
            if not hasattr(self, 'gene_projection'):
                self.gene_projection = nn.Linear(768, 512).to(self.device)
                nn.init.orthogonal_(self.gene_projection.weight)
                nn.init.zeros_(self.gene_projection.bias)
            
            missing_genes = []
            for gene in self.unique_genes:
                emb_file = patient_gene_path / f"{patient_id}_{gene}_embedding.pt"
                if emb_file.exists():
                    try:
                        # Load 768-dimensional embedding
                        embedding = torch.load(emb_file, map_location=self.device, weights_only=True)
                        
                        # Ensure embedding is 2D with shape [1, 768]
                        if embedding.dim() == 1:
                            embedding = embedding.unsqueeze(0)
                        elif embedding.dim() > 2:
                            embedding = embedding.view(1, -1)
                        
                        # Verify 768 dimensions as described in the paper
                        if embedding.size(1) == 768:
                            # Project to 512 dimensions for compatibility
                            with torch.no_grad():
                                embedding = self.gene_projection(embedding)
                        elif embedding.size(1) != 512:
                            if self.verbose:
                                print(f"Warning: Unexpected embedding size for gene {gene}: {embedding.shape}")
                            missing_genes.append(gene)
                            continue
                        
                        embeddings[gene] = embedding
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading embedding for gene {gene}: {str(e)}")
                        missing_genes.append(gene)
                        continue
                else:
                    missing_genes.append(gene)
            
            if not embeddings:
                if self.verbose:
                    print(f"No valid gene embeddings found for patient {patient_id}")
                return None
            
            if self.verbose and missing_genes:
                print(f"Gene embeddings: {len(embeddings)} loaded, {len(missing_genes)} missing")
                
        except Exception as error:
            if self.verbose:
                print(f"Error loading gene embeddings: {str(error)}")
            return None
        
        return embeddings

    def load_structure_embeddings(self, patient_id: str, visit_code: Union[str, int]) -> Optional[Dict[str, torch.Tensor]]:
        """Load 512-d anatomical embeddings per structure for a given visit if available."""
        try:
            embeddings = {}
            formatted_visit = self.format_visit_code(visit_code)
            patient_folder = f"{patient_id}_{formatted_visit}"
            emb_path = self.embeddings_base_path / patient_folder
            if not emb_path.exists():
                return None
            for structure in self.structures:
                pth_file = emb_path / f"{patient_id}_{formatted_visit}_{structure}.pth"
                if pth_file.exists():
                    emb = torch.load(pth_file, map_location=self.device, weights_only=True)
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    elif emb.dim() > 2:
                        emb = emb.view(1, -1)
                    # Project to 512 if needed
                    if emb.size(1) != 512:
                        proj = nn.Linear(emb.size(1), 512).to(self.device)
                        with torch.no_grad():
                            emb = proj(emb)
                    embeddings[structure] = emb.squeeze(0).float()
            return embeddings if embeddings else None
        except Exception:
            return None

    def get_structure_correlations(self, patient_id: str, visit_code: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get BOLD signal correlations for structure-to-structure edges"""
        if self.correlation_df is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        visit_int = int(visit_code)
        
        # Get correlations for this visit
        mask = ((self.correlation_df['patient_id'] == patient_id) & 
                (self.correlation_df['visit_code'] == visit_int))
        visit_corrs = self.correlation_df[mask]
        
        edge_indices = []
        edge_weights = []
        
        # Calculate threshold based on percentile
        if self.bold_threshold == 100:
            threshold = -np.inf  # Include all correlations
        else:
            # Use Pearson correlations above threshold as described in paper
            threshold = np.percentile(visit_corrs['correlation'], 100 - self.bold_threshold)
        
        for _, row in visit_corrs.iterrows():
            correlation = row['correlation']
            
            # Only add edges above threshold (capturing static relationships)
            if abs(correlation) >= threshold:
                struct1 = row['structure1']
                struct2 = row['structure2']
                
                if struct1 in self.node_mapping and struct2 in self.node_mapping:
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

    def get_gene_coexpression_edges(self, available_genes: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get gene-to-gene co-expression patterns as described in the paper"""
        if self.gene_gene_df is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        # Filter to only include genes with available embeddings
        if available_genes is not None:
            available_gene_set = set(available_genes)
        else:
            available_gene_set = set(self.unique_genes)
        
        # Create mapping for available genes only
        available_gene_mapping = {gene: idx for idx, gene in enumerate(available_genes or self.unique_genes) 
                                 if gene in available_gene_set}
        
        edge_indices = []
        edge_weights = []
        
        for _, row in self.gene_gene_df.iterrows():
            gene1, gene2 = row['Gene 1'], row['Gene 2']
            
            # Only add connection if both genes are available
            if (gene1 in available_gene_set and gene2 in available_gene_set and 
                gene1 in available_gene_mapping and gene2 in available_gene_mapping):
                
                gene1_idx = available_gene_mapping[gene1]
                gene2_idx = available_gene_mapping[gene2]
                weight = float(row['Weight'])  # Co-expression strength
                
                # Add bidirectional edges (co-expression is symmetric)
                edge_indices.extend([[gene1_idx, gene2_idx], [gene2_idx, gene1_idx]])
                edge_weights.extend([weight, weight])
            
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        return (
            torch.tensor(edge_indices, dtype=torch.long).t(),
            torch.tensor(edge_weights, dtype=torch.float)
        )

    def get_gene_structure_edges(self, available_genes: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get gene-to-structure mRNA expression levels as described in the paper"""
        if self.gene_structure_df is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            
        edge_indices = []
        edge_weights = []
        
        for _, row in self.gene_structure_df.iterrows():
            gene = row['Gene']
            structure = row['Brain_Structures']
            
            # Check if gene and structure are available
            if (gene in self.gene_mapping and structure in self.node_mapping and
                (available_genes is None or gene in available_genes)):
                
                gene_idx = self.gene_mapping[gene]
                struct_idx = self.node_mapping[structure]
                weight = float(row['Weight'])  # Normalized mRNA expression level
                
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
        """
        Build heterogeneous graph for a single patient following the exact paper description.
        
        This builds the static spatial architecture at any single timepoint as described
        in Stage 2 of the paper.
        """
        try:
            data = HeteroData()
            
            # Get patient visits
            patient_visits = (self.clinical_df[self.clinical_df['BID'] == patient_id]
                            ['VISITCD'].sort_values().unique())
            
            if len(patient_visits) == 0:
                raise ValueError(f"No visits found for patient {patient_id}")
            
            # 1. Build gene nodes (100 AD-associated genes)
            gene_embeddings = self.load_gene_embeddings(patient_id)
            if gene_embeddings is None:
                # Create empty gene features if no embeddings available
                gene_features = torch.zeros((100, 512))  # 100 genes as per paper
                available_genes = []
                if self.verbose:
                    print(f"No gene embeddings for patient {patient_id}, using zero features")
            else:
                # Use only genes with available embeddings
                available_genes = list(gene_embeddings.keys())
                gene_features = torch.stack([
                    gene_embeddings[gene].squeeze(0) for gene in available_genes
                ])
                if self.verbose:
                    print(f"Using {len(available_genes)} gene embeddings for patient {patient_id}")
            
            # Add gene node features to the graph
            data['gene'].x = gene_features
            
            # 2. Build brain structure nodes (32 anatomical regions) with temporal dynamics
            all_structure_features = []
            all_spatial_edges = []
            all_temporal_edges = []
            node_offset = 0
            
            for visit_idx, visit in enumerate(patient_visits):
                visit_features = []
                
                # Load anatomical embeddings if available
                anat_embeddings = self.load_structure_embeddings(patient_id, visit)
                
                # Get 107-dimensional radiomic features for each structure
                rad_list = []
                for structure in self.structures:
                    radiomic_features = self.get_radiomics_features(patient_id, visit, structure)
                    if radiomic_features is None:
                        radiomic_features = torch.zeros(107)
                    rad_list.append(radiomic_features)
                
                # Convert 107-dim radiomic features to 512-dim for consistency
                # This projects the radiomic features to the same space as gene embeddings
                if not hasattr(self, 'radiomic_projection'):
                    self.radiomic_projection = nn.Linear(107, 512).to(self.device)
                    nn.init.xavier_uniform_(self.radiomic_projection.weight)
                    nn.init.zeros_(self.radiomic_projection.bias)
                
                # Project radiomic features to 512 dimensions
                visit_rad_tensor = torch.stack(rad_list)
                with torch.no_grad():
                    projected_rad = self.radiomic_projection(visit_rad_tensor)
                
                # Concatenate anatomical 512 (if present) with projected radiomics 512 => 1024 per node
                fused_visit = []
                for i, structure in enumerate(self.structures):
                    rad_512 = projected_rad[i]
                    if anat_embeddings is not None and structure in anat_embeddings:
                        anat_512 = anat_embeddings[structure]
                    else:
                        anat_512 = torch.zeros(512, dtype=torch.float, device=rad_512.device)
                    fused_visit.append(torch.cat([anat_512, rad_512], dim=0))
                fused_visit_tensor = torch.stack(fused_visit)  # [32, 1024]
                all_structure_features.append(fused_visit_tensor)
                
                # 3. Add structure-to-structure edges (BOLD correlations)
                edge_index, edge_weights = self.get_structure_correlations(patient_id, visit)
                if edge_index.size(1) > 0:
                    edge_index = edge_index + node_offset
                    all_spatial_edges.append((edge_index, edge_weights))
                
                # 4. Add temporal connections between consecutive visits with radiomics delta weights
                if visit_idx > 0:
                    prev_visit = patient_visits[visit_idx - 1]
                    temp_edges = []
                    temp_attrs = []  # 107-d delta vectors per edge
                    
                    for struct_idx, structure in enumerate(self.structures):
                        prev_node = node_offset - len(self.structures) + struct_idx
                        curr_node = node_offset + struct_idx
                        
                        # Temporal edge from previous to current visit (unidirectional)
                        temp_edges.append([prev_node, curr_node])
                        
                        # Compute radiomics delta (curr - prev) for this structure
                        prev_feat = self.get_radiomics_features(patient_id, prev_visit, structure)
                        curr_feat = self.get_radiomics_features(patient_id, visit, structure)
                        if prev_feat is not None and curr_feat is not None:
                            # Ensure tensors are 1D with length 107
                            prev_vec = prev_feat.view(-1)[:107]
                            curr_vec = curr_feat.view(-1)[:107]
                            if prev_vec.numel() < 107:
                                prev_vec = torch.nn.functional.pad(prev_vec, (0, 107 - prev_vec.numel()))
                            if curr_vec.numel() < 107:
                                curr_vec = torch.nn.functional.pad(curr_vec, (0, 107 - curr_vec.numel()))
                            delta_vec = (curr_vec - prev_vec).float()
                        else:
                            delta_vec = torch.zeros(107, dtype=torch.float)
                        temp_attrs.append(delta_vec)
                    
                    if temp_edges:
                        temp_edge_index = torch.tensor(temp_edges, dtype=torch.long).t()
                        temp_edge_attr = torch.stack(temp_attrs)  # [E, 107]
                        # Store under per-transition relation name for clarity
                        rel_name = f"temporal_v{visit_idx-1}_to_v{visit_idx}"
                        data['structure', rel_name, 'structure'].edge_index = temp_edge_index
                        data['structure', rel_name, 'structure'].edge_attr = temp_edge_attr
                
                node_offset += len(self.structures)
            
            # Combine all structure features
            if all_structure_features:
                data['structure'].x = torch.cat(all_structure_features, dim=0)
            
            # Add spatial edges (BOLD correlations)
            if all_spatial_edges:
                all_spatial_indices = torch.cat([e[0] for e in all_spatial_edges], dim=1)
                all_spatial_weights = torch.cat([e[1] for e in all_spatial_edges])
                data['structure', 'bold_correlated', 'structure'].edge_index = all_spatial_indices
                data['structure', 'bold_correlated', 'structure'].edge_attr = all_spatial_weights.view(-1, 1)
            
            # Note: temporal edges are stored per-transition above with 107-d attributes
            
            # 5. Add gene-to-gene edges (co-expression patterns)
            gene_edge_index, gene_edge_weights = self.get_gene_coexpression_edges(available_genes)
            if gene_edge_index.size(1) > 0:
                data['gene', 'coexpressed_with', 'gene'].edge_index = gene_edge_index
                data['gene', 'coexpressed_with', 'gene'].edge_attr = gene_edge_weights.view(-1, 1)
            
            # 6. Add gene-to-structure edges (mRNA expression levels)
            gene_struct_edge_index, gene_struct_weights = self.get_gene_structure_edges(available_genes)
            if gene_struct_edge_index.size(1) > 0:
                data['gene', 'expressed_in', 'structure'].edge_index = gene_struct_edge_index
                data['gene', 'expressed_in', 'structure'].edge_attr = gene_struct_weights.view(-1, 1)
            
            # Add metadata
            data.patient_id = patient_id
            data.visits = patient_visits.tolist()
            
            # Add survival information (per visit)
            times = []
            events = []
            pacc_scores = []
            for visit in patient_visits:
                survival_info = self.get_patient_survival_info(patient_id, visit)
                if survival_info:
                    times.append(survival_info.time)
                    events.append(survival_info.event)
                    pacc_scores.append(survival_info.pacc_scores)
                    
            data.survival_times = torch.tensor(times)
            data.events = torch.tensor(events)
            data.pacc_scores = torch.tensor(pacc_scores)
            
            if self.verbose:
                print(f"Built graph for patient {patient_id}:")
                print(f"  - Structure nodes: {data['structure'].x.size(0)}")
                print(f"  - Gene nodes: {data['gene'].x.size(0)}")
                print(f"  - Visits: {len(patient_visits)}")
                if ('structure', 'bold_correlated', 'structure') in data.edge_types:
                    print(f"  - BOLD correlation edges: {data['structure', 'bold_correlated', 'structure'].edge_index.size(1)}")
                if ('gene', 'coexpressed_with', 'gene') in data.edge_types:
                    print(f"  - Gene co-expression edges: {data['gene', 'coexpressed_with', 'gene'].edge_index.size(1)}")
                if ('gene', 'expressed_in', 'structure') in data.edge_types:
                    print(f"  - Gene-structure edges: {data['gene', 'expressed_in', 'structure'].edge_index.size(1)}")
            
            return data
            
        except Exception as error:
            if self.verbose:
                print(f"Error building graph for patient {patient_id}: {str(error)}")
                traceback.print_exc()
            raise
