import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class CommunityGuidedGAT(nn.Module):
    def __init__(self, feature_dim, num_communities, hidden_dim, embed_dim, ablate=None):
        super(CommunityGuidedGAT, self).__init__()

        self.ablate = ablate

        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.gat1 = GATConv(hidden_dim*2, hidden_dim, heads=8, concat=True)
        self.gat2 = GATConv(hidden_dim * 8, embed_dim, heads=1, concat=False)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, community_ids, community_ids_fine=None, edge_attr=None):

        h_feat = self.feat_encoder(x)

        num_current_communities = community_ids.max().item() + 1

        h_comm_sum = torch.zeros(
            (num_current_communities, h_feat.size(1)), 
            device=h_feat.device
        ).scatter_add_(
            0, 
            community_ids.unsqueeze(1).repeat(1, h_feat.size(1)), 
            h_feat
        )

        comm_counts = torch.zeros(
            num_current_communities, 
            device=h_feat.device, 
            dtype=h_feat.dtype
        ).scatter_add_(
            0, 
            community_ids, 
            torch.ones_like(community_ids, dtype=h_feat.dtype)
        )

        comm_counts = comm_counts.clamp(min=1).unsqueeze(1)
        h_comm_mean = h_comm_sum / comm_counts

        h_comm = h_comm_mean[community_ids]

        if self.ablate == 'cci':
            h = torch.cat([h_feat, h_feat], dim=1)
        else:
            h = torch.cat([h_feat, h_comm], dim=1)

        src, dst = edge_index

        mask_ids = community_ids_fine if community_ids_fine is not None else community_ids
        src_comm = mask_ids[src]
        dst_comm = mask_ids[dst]

        intra_community_mask = (src_comm == dst_comm)

        if self.ablate == 'iim':
            edge_index_intra = edge_index
        else:
            edge_index_intra = edge_index[:, intra_community_mask]

        h = self.gat1(h, edge_index_intra)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)

        z_node = self.gat2(h, edge_index_intra)

        z_proj = self.projector(z_node)

        return z_node, z_proj