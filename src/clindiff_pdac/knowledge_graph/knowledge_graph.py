"""
Pancreatic Cancer Clinical Knowledge Graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import json


class PDACKnowledgeGraph:
    """
    Clinical Knowledge Graph for Pancreatic Ductal Adenocarcinoma (PDAC)
    Encodes relationships between symptoms, lab values, treatments, and outcomes
    """
    
    def __init__(self):
        self.entities = self._initialize_entities()
        self.relations = self._initialize_relations()
        self.constraints = self._initialize_constraints()
        
    def _initialize_entities(self) -> Dict[str, Set[str]]:
        """Initialize entity types and their instances"""
        return {
            'symptoms': {
                'jaundice', 'abdominal_pain', 'weight_loss', 'nausea',
                'back_pain', 'bloating', 'diarrhea', 'fatigue', 'anorexia',
                'vomiting', 'fever', 'pruritus'
            },
            'laboratory': {
                'CA19-9', 'CEA', 'CA125', 'bilirubin_total', 'bilirubin_direct',
                'ALT', 'AST', 'ALP', 'GGT', 'amylase', 'lipase',
                'glucose', 'HbA1c', 'albumin', 'hemoglobin', 'WBC', 'platelets',
                'creatinine', 'BUN', 'CRP'
            },
            'imaging': {
                'CT_pancreas', 'MRI_pancreas', 'PET_CT', 'EUS', 'ERCP',
                'liver_metastasis', 'lung_metastasis', 'peritoneal_metastasis',
                'lymphadenopathy', 'vascular_invasion'
            },
            'treatments': {
                'surgery_whipple', 'surgery_distal', 'surgery_total',
                'chemotherapy_gem', 'chemotherapy_folfirinox', 'chemotherapy_ag',
                'radiation', 'palliative_care', 'neoadjuvant', 'adjuvant'
            },
            'staging': {
                'T1', 'T2', 'T3', 'T4', 'N0', 'N1', 'M0', 'M1',
                'stage_I', 'stage_II', 'stage_III', 'stage_IV'
            },
            'comorbidities': {
                'diabetes', 'chronic_pancreatitis', 'biliary_obstruction',
                'cholangitis', 'peptic_ulcer', 'smoking', 'alcohol',
                'obesity', 'family_history_pancreatic'
            }
        }
    
    def _initialize_relations(self) -> List[Dict]:
        """
        Initialize clinical relations with strength weights
        Format: {'from': entity, 'to': entity, 'relation': type, 'strength': float}
        """
        return [
            # CA19-9 relations (most important PDAC biomarker)
            {'from': 'CA19-9', 'relation': 'correlates_with', 'to': 'stage_IV', 'strength': 0.85},
            {'from': 'CA19-9', 'relation': 'correlates_with', 'to': 'liver_metastasis', 'strength': 0.80},
            {'from': 'CA19-9', 'relation': 'elevated_in', 'to': 'pancreatic_cancer', 'strength': 0.90},
            {'from': 'CA19-9', 'relation': 'monitored_during', 'to': 'chemotherapy_gem', 'strength': 0.75},
            
            # Bilirubin relations (jaundice pathway)
            {'from': 'bilirubin_total', 'relation': 'causes', 'to': 'jaundice', 'strength': 0.95},
            {'from': 'bilirubin_total', 'relation': 'elevated_in', 'to': 'biliary_obstruction', 'strength': 0.90},
            {'from': 'biliary_obstruction', 'relation': 'causes', 'to': 'jaundice', 'strength': 0.85},
            {'from': 'biliary_obstruction', 'relation': 'indicates', 'to': 'CT_pancreas', 'strength': 0.80},
            
            # Symptoms relations
            {'from': 'weight_loss', 'relation': 'correlates_with', 'to': 'advanced_stage', 'strength': 0.70},
            {'from': 'abdominal_pain', 'relation': 'located_in', 'to': 'pancreas', 'strength': 0.75},
            {'from': 'back_pain', 'relation': 'suggests', 'to': 'T4', 'strength': 0.65},
            {'from': 'jaundice', 'relation': 'suggests', 'to': 'head_lesion', 'strength': 0.85},
            
            # Diabetes relations
            {'from': 'diabetes', 'relation': 'risk_factor', 'to': 'pancreatic_cancer', 'strength': 0.60},
            {'from': 'new_onset_diabetes', 'relation': 'paraneoplastic', 'to': 'pancreatic_cancer', 'strength': 0.70},
            {'from': 'glucose', 'relation': 'correlates_with', 'to': 'diabetes', 'strength': 0.85},
            
            # Treatment relations
            {'from': 'stage_I', 'relation': 'eligible_for', 'to': 'surgery_whipple', 'strength': 0.95},
            {'from': 'stage_II', 'relation': 'eligible_for', 'to': 'surgery_distal', 'strength': 0.90},
            {'from': 'stage_III', 'relation': 'may_receive', 'to': 'neoadjuvant', 'strength': 0.85},
            {'from': 'stage_IV', 'relation': 'contraindicates', 'to': 'surgery_whipple', 'strength': 0.95},
            {'from': 'T4', 'relation': 'may_require', 'to': 'neoadjuvant', 'strength': 0.80},
            
            # Metastasis relations
            {'from': 'liver_metastasis', 'relation': 'indicates', 'to': 'M1', 'strength': 0.98},
            {'from': 'lung_metastasis', 'relation': 'indicates', 'to': 'M1', 'strength': 0.95},
            {'from': 'peritoneal_metastasis', 'relation': 'indicates', 'to': 'M1', 'strength': 0.95},
            {'from': 'CA19-9', 'relation': 'correlates_with', 'to': 'metastatic_burden', 'strength': 0.75},
            
            # Age and comorbidity relations
            {'from': 'age', 'relation': 'influences', 'to': 'treatment_choice', 'strength': 0.60},
            {'from': 'albumin', 'relation': 'predicts', 'to': 'surgical_outcome', 'strength': 0.70},
            {'from': 'CRP', 'relation': 'marker_of', 'to': 'inflammation', 'strength': 0.80},
            
            # Liver function
            {'from': 'ALT', 'relation': 'correlates_with', 'to': 'AST', 'strength': 0.85},
            {'from': 'ALP', 'relation': 'elevated_in', 'to': 'biliary_obstruction', 'strength': 0.80},
            {'from': 'GGT', 'relation': 'marker_of', 'to': 'biliary_damage', 'strength': 0.75},
            
            # Treatment response
            {'from': 'CA19-9', 'relation': 'declines_with', 'to': 'effective_chemotherapy', 'strength': 0.80},
            {'from': 'CEA', 'relation': 'monitored_with', 'to': 'metastatic_disease', 'strength': 0.65},
        ]
    
    def _initialize_constraints(self) -> Dict[str, Dict]:
        """
        Initialize physiological constraints for values
        Format: {'variable': {'min': float, 'max': float, 'unit': str}}
        """
        return {
            'CA19-9': {'min': 0, 'max': 10000, 'unit': 'U/mL', 'normal_max': 37},
            'CEA': {'min': 0, 'max': 1000, 'unit': 'ng/mL', 'normal_max': 5},
            'CA125': {'min': 0, 'max': 5000, 'unit': 'U/mL', 'normal_max': 35},
            'bilirubin_total': {'min': 0, 'max': 30, 'unit': 'mg/dL', 'normal_max': 1.2},
            'bilirubin_direct': {'min': 0, 'max': 20, 'unit': 'mg/dL', 'normal_max': 0.3},
            'ALT': {'min': 0, 'max': 1000, 'unit': 'U/L', 'normal_max': 40},
            'AST': {'min': 0, 'max': 1000, 'unit': 'U/L', 'normal_max': 40},
            'ALP': {'min': 0, 'max': 2000, 'unit': 'U/L', 'normal_max': 120},
            'GGT': {'min': 0, 'max': 1000, 'unit': 'U/L', 'normal_max': 50},
            'amylase': {'min': 0, 'max': 2000, 'unit': 'U/L', 'normal_max': 125},
            'lipase': {'min': 0, 'max': 5000, 'unit': 'U/L', 'normal_max': 60},
            'glucose': {'min': 40, 'max': 500, 'unit': 'mg/dL', 'normal_max': 100},
            'HbA1c': {'min': 4, 'max': 18, 'unit': '%', 'normal_max': 5.7},
            'albumin': {'min': 1, 'max': 6, 'unit': 'g/dL', 'normal_min': 3.5},
            'hemoglobin': {'min': 3, 'max': 20, 'unit': 'g/dL', 'normal_min': 12},
            'platelets': {'min': 20, 'max': 1000, 'unit': '10^3/uL', 'normal_min': 150},
            'creatinine': {'min': 0.3, 'max': 20, 'unit': 'mg/dL', 'normal_max': 1.2},
            'CRP': {'min': 0, 'max': 500, 'unit': 'mg/L', 'normal_max': 3},
            'age': {'min': 18, 'max': 100, 'unit': 'years'},
        }
    
    def get_neighbors(self, entity: str, relation_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get neighboring entities and their relation strengths
        
        Args:
            entity: Source entity
            relation_type: Optional filter for relation type
            
        Returns:
            List of (neighbor_entity, strength) tuples
        """
        neighbors = []
        for rel in self.relations:
            if rel['from'] == entity:
                if relation_type is None or rel['relation'] == relation_type:
                    neighbors.append((rel['to'], rel['strength']))
        return neighbors
    
    def check_constraint(self, variable: str, value: float) -> bool:
        """
        Check if a value is within physiological range
        
        Args:
            variable: Variable name
            value: Value to check
            
        Returns:
            True if within range
        """
        if variable not in self.constraints:
            return True  # No constraint defined
        
        constraint = self.constraints[variable]
        if 'min' in constraint and value < constraint['min']:
            return False
        if 'max' in constraint and value > constraint['max']:
            return False
        return True
    
    def get_relation_path(self, entity1: str, entity2: str, max_hops: int = 3) -> List[List[str]]:
        """
        Find relation paths between two entities
        
        Args:
            entity1: Start entity
            entity2: Target entity
            max_hops: Maximum path length
            
        Returns:
            List of paths (each path is a list of entities)
        """
        def dfs(current: str, target: str, path: List[str], hops: int, paths: List[List[str]]):
            if hops > max_hops:
                return
            if current == target and len(path) > 1:
                paths.append(path.copy())
                return
            
            for neighbor, _ in self.get_neighbors(current):
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, target, path, hops + 1, paths)
                    path.pop()
        
        paths = []
        dfs(entity1, entity2, [entity1], 0, paths)
        return paths
    
    def get_implication_rules(self, antecedent: str) -> List[Dict]:
        """
        Get rules where antecedent implies certain conclusions
        
        Args:
            antecedent: Entity that is known/observed
            
        Returns:
            List of implication rules
        """
        implications = []
        for rel in self.relations:
            if rel['from'] == antecedent and rel['relation'] in ['causes', 'indicates', 'suggests', 'elevated_in']:
                implications.append({
                    'conclusion': rel['to'],
                    'relation': rel['relation'],
                    'confidence': rel['strength']
                })
        return implications


class KnowledgeGraphEncoder(nn.Module):
    """
    Neural network encoder for the PDAC knowledge graph
    """
    
    def __init__(self, kg: PDACKnowledgeGraph, embedding_dim: int = 256):
        super().__init__()
        self.kg = kg
        self.embedding_dim = embedding_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.ModuleDict()
        for entity_type, entities in kg.entities.items():
            self.entity_embeddings[entity_type] = nn.Embedding(
                len(entities), 
                embedding_dim // 2
            )
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(10, embedding_dim // 2)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(embedding_dim, embedding_dim),
            GraphConvLayer(embedding_dim, embedding_dim),
            GraphConvLayer(embedding_dim, embedding_dim)
        ])
        
        # Output projection
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, entity_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode entities into embeddings
        
        Args:
            entity_ids: Dict of {entity_type: tensor of indices}
            
        Returns:
            [batch, embedding_dim] embeddings
        """
        embeddings = []
        
        for entity_type, ids in entity_ids.items():
            if entity_type in self.entity_embeddings:
                emb = self.entity_embeddings[entity_type](ids)
                embeddings.append(emb)
        
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            if combined.shape[-1] > self.embedding_dim:
                combined = combined[:, :self.embedding_dim]
            elif combined.shape[-1] < self.embedding_dim:
                padding = torch.zeros(
                    *combined.shape[:-1], 
                    self.embedding_dim - combined.shape[-1],
                    device=combined.device
                )
                combined = torch.cat([combined, padding], dim=-1)
        else:
            combined = torch.zeros(entity_ids[list(entity_ids.keys())[0]].shape[0], self.embedding_dim)
        
        return self.projection(combined)


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, num_nodes, in_dim]
            adj: [num_nodes, num_nodes] adjacency matrix (optional)
        """
        if adj is None:
            # Average pooling if no adjacency
            x = x.mean(dim=1)
        else:
            # Graph convolution
            x = torch.bmm(adj, x)
        return self.norm(F.relu(self.linear(x)))
