import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, mask_feature

class HCGTrainer:
    def __init__(self, model, optimizer, lambda_nn=1.0, lambda_nc=0.5, temperature=0.5, ablate=None, pe=0.3, pf=0.3):
        self.model = model
        self.optimizer = optimizer
        self.temp = temperature
        self.lambda_nn = lambda_nn
        self.lambda_nc = lambda_nc
        self.ablate = ablate
        self.pe = pe
        self.pf = pf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def augment_graph(self, data):

        edge_index_1, _ = dropout_edge(data.edge_index, p=self.pe, force_undirected=True)
        x_1 = data.x.clone()

        x_2, _ = mask_feature(data.x, p=self.pf, mode='col')
        edge_index_2 = data.edge_index.clone()

        return (x_1, edge_index_1), (x_2, edge_index_2)

    def info_nce_loss_back(self, z1, z2):

        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        N = z1.shape[0]
        block_size=1024

        sim_pos = (z1 * z2).sum(dim=1)
        positives = torch.cat([sim_pos, sim_pos], dim=0)

        all_repr = torch.cat([z1, z2], dim=0)
        total_size = all_repr.shape[0]

        negatives = []

        for block in torch.split(all_repr, block_size, dim=0):
            B = block.shape[0]

            sim_block = F.cosine_similarity(
                block.unsqueeze(1),
                all_repr.unsqueeze(0),
                dim=2
            )

            start_idx = len(negatives) * block_size
            end_idx = start_idx + B

            block_indices = torch.arange(start_idx, end_idx, device=self.device)

            sim_block[torch.arange(B), block_indices] = -9e15

            neg_block = torch.logsumexp(sim_block / self.temp, dim=1)
            negatives.append(neg_block)

        negatives = torch.cat(negatives)

        loss = - (positives / self.temp - negatives).mean()
        return loss

    def info_nce_loss(self, z1, z2):

        z1 = F.normalize(z1, dim=1) 
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        device = z1.device

        Z = torch.cat([z1, z2], dim=0)

        S = torch.matmul(Z, Z.T)

        logits = S / self.temp

        positive_mask = torch.zeros((2 * N, 2 * N), device=device, dtype=torch.bool)
        positive_mask[0:N, N:2*N] = torch.eye(N, device=device, dtype=torch.bool)
        positive_mask[N:2*N, 0:N] = torch.eye(N, device=device, dtype=torch.bool)

        positive_logits = logits[positive_mask].view(2 * N, -1)

        diag_mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        logits.masked_fill_(diag_mask, -9e15) 

        log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)

        loss = - (positive_logits - log_sum_exp).mean()
        return loss

    def calculate_node_community_loss(self, z_proj, community_ids, flow_mask, train_mask):

        z_train = z_proj[train_mask]
        comm_ids_train = community_ids[train_mask]

        if z_train.size(0) < 2:
             return torch.tensor(0.0, device=self.device)

        z_norm = F.normalize(z_proj, dim=1) 
        unique_comm_ids = community_ids.unique()
        num_communities = len(unique_comm_ids)

        map_tensor = torch.arange(num_communities, device=self.device)
        comm_id_map = torch.empty((community_ids.max().item() + 1,), 
                                  dtype=torch.long, 
                                  device=self.device).fill_(-1)
        comm_id_map[unique_comm_ids] = map_tensor
        scatter_index = comm_id_map[community_ids] 

        comm_sum = torch.zeros((num_communities, z_norm.size(1)), device=self.device)
        comm_sum.scatter_add_(0, scatter_index.unsqueeze(1).expand_as(z_norm), z_norm)

        comm_counts = torch.zeros(num_communities, 1, device=self.device)
        comm_counts.scatter_add_(0, scatter_index.unsqueeze(1), torch.ones_like(z_norm[:, :1]))
        m_k = comm_sum / (comm_counts + 1e-6) 

        m_k_norm = F.normalize(m_k, dim=1)

        z_train_norm = F.normalize(z_train, dim=1)
        similarity_matrix = torch.matmul(z_train_norm, m_k_norm.transpose(0, 1)) / self.temp

        pos_indices = comm_id_map[comm_ids_train].to(self.device)

        loss_nc = F.cross_entropy(similarity_matrix, pos_indices)

        return loss_nc

    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        train_mask = (data.y == 0)

        if train_mask.sum() == 0:
            return 0.0

        (x1, ei1), (x2, ei2) = self.augment_graph(data)

        _fine = getattr(data, 'community_id_fine', None)
        _, z1_proj = self.model(x1, ei1, data.community_id, _fine)
        _, z2_proj = self.model(x2, ei2, data.community_id, _fine)

        z1_train = z1_proj[train_mask]
        z2_train = z2_proj[train_mask]

        L_nn = self.info_nce_loss(z1_train, z2_train)

        L_nc = self.calculate_node_community_loss(z1_proj, data.community_id, None, train_mask)

        if self.ablate == 'lnn':
            loss = self.lambda_nc * L_nc
        elif self.ablate == 'lnc':
            loss = self.lambda_nn * L_nn
        else:
            loss = self.lambda_nn * L_nn + self.lambda_nc * L_nc
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_epoch(self, data):
        self.model.eval()

        with torch.no_grad():

            val_mask = (data.y == 0) 

            if val_mask.sum() == 0:
                return 0.0

            (x1, ei1), (x2, ei2) = self.augment_graph(data)

            _fine = getattr(data, 'community_id_fine', None)
            _, z1_proj = self.model(x1, ei1, data.community_id, _fine)
            _, z2_proj = self.model(x2, ei2, data.community_id, _fine)

            z1_val = z1_proj[val_mask]
            z2_val = z2_proj[val_mask]

            L_nn = self.info_nce_loss(z1_val, z2_val)

            L_nc = self.calculate_node_community_loss(z1_proj, data.community_id, None, val_mask)

            if self.ablate == 'lnn':
                total_loss = self.lambda_nc * L_nc
            elif self.ablate == 'lnc':
                total_loss = self.lambda_nn * L_nn
            else:
                total_loss = self.lambda_nn * L_nn + self.lambda_nc * L_nc
        return total_loss.item()    