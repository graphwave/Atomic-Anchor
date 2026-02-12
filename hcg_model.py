import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class CommunityGuidedGAT(nn.Module):
    def __init__(self, feature_dim, num_communities, hidden_dim, embed_dim):
        super(CommunityGuidedGAT, self).__init__()
        
        # 1. Feature encoding layer
        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Community embedding layer (Learnable Community Embeddings)
        #self.comm_embedding = nn.Embedding(num_communities, hidden_dim) # nn.Embedding: Lookup Table
        # 2. [Change 2] Add dynamic community encoder (Community Encoder)
        # Input dimension is 1 (community ID scalar)
        #self.comm_encoder = nn.Sequential(
        #    nn.Linear(1, hidden_dim), # Map scalar ID to hidden dimension
        #    nn.BatchNorm1d(hidden_dim),
        #    nn.ReLU()
        #)
        # 3. Graph Attention Layer (GAT)
        # Input dimension is hidden_dim * 2 (feature + community)
        self.gat1 = GATConv(hidden_dim*2, hidden_dim, heads=8, concat=True)
        self.gat2 = GATConv(hidden_dim * 8, embed_dim, heads=1, concat=False)
        
        # 4. Projection Head (for Contrastive Learning)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, community_ids, edge_attr=None):
        # --- Feature Fusion ---
        h_feat = self.feat_encoder(x)
        
        # ... (Community Feature Statistical Aggregation (CFS) part of the code remains unchanged) ...
        # 1. Dynamically get the total number of current communities
        num_current_communities = community_ids.max().item() + 1
        # 2. Calculate Sum
        h_comm_sum = torch.zeros(
            (num_current_communities, h_feat.size(1)), 
            device=h_feat.device
        ).scatter_add_(
            0, 
            community_ids.unsqueeze(1).repeat(1, h_feat.size(1)), 
            h_feat
        )
        # 3. Calculate Count
        comm_counts = torch.zeros(
            num_current_communities, 
            device=h_feat.device, 
            dtype=h_feat.dtype
        ).scatter_add_(
            0, 
            community_ids, 
            torch.ones_like(community_ids, dtype=h_feat.dtype)
        )
        # 4. Calculate Mean
        comm_counts = comm_counts.clamp(min=1).unsqueeze(1)
        h_comm_mean = h_comm_sum / comm_counts
        # 5. Map back to nodes
        h_comm = h_comm_mean[community_ids]
        
        # Concatenate features
        h = torch.cat([h_feat, h_comm], dim=1)
        #20251227 Ablation Experiment: Test experimental effect when Community Feature Injection (CCI) is excluded
        #h = torch.cat([h_feat, h_feat], dim=1)
        # --------------------- [Core Modification: Intra-Community Edge Filtering] ---------------------
        # Purpose: Weaken or even eliminate the influence of non-local community neighbors
        # Logic: Only retain edges where the source and target nodes have the same community_id
        
        src, dst = edge_index
        # Get the community IDs of the two nodes of the edge
        src_comm = community_ids[src]
        dst_comm = community_ids[dst]
        
        # Generate mask: True only when the community IDs of both ends are the same
        intra_community_mask = (src_comm == dst_comm)
        
        # Apply mask to filter out "intra-community edges"
        # Note: This will discard physical connections across communities (e.g., a Host is connected to a Flow belonging to a different community)
        edge_index_intra = edge_index[:, intra_community_mask]
        #2025-12-27 Ablation Experiment
        #edge_index_intra = edge_index
        # If you need to retain part of the cross-community information but weaken it, you can avoid complete filtering,
        # and instead pass edge_weight to GAT (need to modify GATConv parameters), but direct filtering is the most thorough.
        
        # --------------------------------------------------------------------

        # --- Graph Encoding ---
        # Perform message passing using the filtered edge_index_intra
        
        # First layer of GAT
        h = self.gat1(h, edge_index_intra) # Use intra-community edges
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        # Second layer of GAT -> Node embeddings
        # Also use filtered edges to prevent contamination from second-order neighbors
        z_node = self.gat2(h, edge_index_intra)
        
        # --- Contrastive Projection ---
        z_proj = self.projector(z_node)
        
        return z_node, z_proj