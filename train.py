import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, mask_feature

class HCGTrainer:
    def __init__(self, model, optimizer, lambda_nn=1.0, lambda_nc=0.5, temperature=0.5):
        """
        Initialize HCGTrainer, add weights for Node-Node (NN) and Node-Community (NC) losses
        """
        self.model = model
        self.optimizer = optimizer
        self.temp = temperature
        self.lambda_nn = lambda_nn
        self.lambda_nc = lambda_nc
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def augment_graph(self, data):
        """
        Graph data augmentation: generate two views (Edge Dropping and Feature Masking)
        """
        # View 1: Drop Edges pe
        edge_index_1, _ = dropout_edge(data.edge_index, p=0.3, force_undirected=True)
        x_1 = data.x.clone()
        
        # View 2: Mask Featurespf
        x_2, _ = mask_feature(data.x, p=0.3, mode='col')
        edge_index_2 = data.edge_index.clone()
        
        return (x_1, edge_index_1), (x_2, edge_index_2)

    def info_nce_loss_back(self, z1, z2):
        # Normalization (consistent with original logic)
        z1 = F.normalize(z1, dim=1, p=2)  # [N, D]
        z2 = F.normalize(z2, dim=1, p=2)  # [N, D]
        N = z1.shape[0]
        block_size=1024
        # -------------------------- 1. Calculate positive sample similarity (vectorized) --------------------------
        # Positive sample pairs: z1[i]↔z2[i], z2[i]↔z1[i], dot product equals cosine similarity (normalized)
        sim_pos = (z1 * z2).sum(dim=1)  # [N,]
        positives = torch.cat([sim_pos, sim_pos], dim=0)  # [2N,]
        
        # -------------------------- 2. Negative sample pool (consistent with original logic) --------------------------
        all_repr = torch.cat([z1, z2], dim=0)  # [2N, D]
        total_size = all_repr.shape[0]  # 2N
        
        # -------------------------- 3. Block-wise calculation of negative sample logsumexp (parallelization) --------------------------
        # Block idea: Split all_repr into multiple blocks, calculate similarity between each block and full all_repr ([B, 2N])
        # Utilize GPU matrix multiplication parallelism, no Python loops, maximize speed
        negatives = []
        # Split all_repr by block_size (dimension 0)
        for block in torch.split(all_repr, block_size, dim=0):
            B = block.shape[0]  # Number of samples in current block
            
            # Calculate similarity matrix between current block and full all_repr [B, 2N] (GPU parallel computation)
            sim_block = F.cosine_similarity(
                block.unsqueeze(1),  # [B, 1, D]
                all_repr.unsqueeze(0),  # [1, 2N, D]
                dim=2
            )  # [B, 2N]
            
            # Efficient mask construction: mask self-similarity (replace manual loop)
            # Calculate start/end indices of current block
            start_idx = len(negatives) * block_size
            end_idx = start_idx + B
            # Generate indices of samples in the block: [start_idx, start_idx+1, ..., end_idx-1]
            block_indices = torch.arange(start_idx, end_idx, device=self.device)
            # Broadcast to construct mask, set sim_block[i][block_indices[i]] to a very small value
            sim_block[torch.arange(B), block_indices] = -9e15
            
            # Calculate logsumexp (parallelization)
            neg_block = torch.logsumexp(sim_block / self.temp, dim=1)  # [B,]
            negatives.append(neg_block)
        
        negatives = torch.cat(negatives)  # [2N,]
        
        # -------------------------- 4. Calculate final Loss --------------------------
        loss = - (positives / self.temp - negatives).mean()
        return loss

    def info_nce_loss(self, z1, z2):
        # Normalization
        z1 = F.normalize(z1, dim=1) 
        z2 = F.normalize(z2, dim=1)
        
        N = z1.size(0)
        device = z1.device
        
        # Concatenate all samples as negative sample pool Z [2N, D]
        Z = torch.cat([z1, z2], dim=0)

        # 1. Calculate all similarity matrices (Dot Product = Cosine Similarity)
        # S: [2N, 2N]
        S = torch.matmul(Z, Z.T) # Only one matrix multiplication
        
        # 2. Convert to logits
        logits = S / self.temp
        
        # 3. Identify positive sample pairs
        # Positive Mask: (z1[i], z2[i]) and (z2[i], z1[i])
        positive_mask = torch.zeros((2 * N, 2 * N), device=device, dtype=torch.bool)
        positive_mask[0:N, N:2*N] = torch.eye(N, device=device, dtype=torch.bool)
        positive_mask[N:2*N, 0:N] = torch.eye(N, device=device, dtype=torch.bool)
        
        # Positive Logits
        positive_logits = logits[positive_mask].view(2 * N, -1) # [2N, 1]
        
        # 4. Mask self-similarity (Negative Mask)
        # Set main diagonal (i vs i) to a very small value
        diag_mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        logits.masked_fill_(diag_mask, -9e15) 
        
        # 5. Calculate Loss (Log-Sum-Exp)
        # Since we only mask self-similarity, logsumexp includes all positive and negative samples
        log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True) # [2N, 1]
        
        # Loss = - (positive_logit - logsumexp(all_logits))
        loss = - (positive_logits - log_sum_exp).mean()
        return loss

    def calculate_node_community_loss(self, z_proj, community_ids, flow_mask, train_mask):
        """
        Calculate Node-Community Contrastive Loss (L_NC)
        """
        # --- 1. Filter training set node embeddings and community IDs ---
        
        # Only use training set (Benign Flow) nodes for positive sample calculation
        # Note: train_mask here must be Benign nodes among Flow nodes
        z_train = z_proj[train_mask]
        comm_ids_train = community_ids[train_mask]

        if z_train.size(0) < 2:
             return torch.tensor(0.0, device=self.device)

        # --- 2. Dynamically calculate community centers (m_k) ---
        
        # Use community_ids and normalized z_proj of the entire graph to calculate all community centers
        z_norm = F.normalize(z_proj, dim=1) 
        unique_comm_ids = community_ids.unique()
        num_communities = len(unique_comm_ids)

        # Map original community IDs to continuous indices (0 to N-1)
        map_tensor = torch.arange(num_communities, device=self.device)
        comm_id_map = torch.empty((community_ids.max().item() + 1,), 
                                  dtype=torch.long, 
                                  device=self.device).fill_(-1)
        comm_id_map[unique_comm_ids] = map_tensor
        scatter_index = comm_id_map[community_ids] 

        # Efficiently calculate the sum of community embeddings using scatter_add_ (Sum)
        comm_sum = torch.zeros((num_communities, z_norm.size(1)), device=self.device)
        comm_sum.scatter_add_(0, scatter_index.unsqueeze(1).expand_as(z_norm), z_norm)
        
        # Calculate community centers m_k
        comm_counts = torch.zeros(num_communities, 1, device=self.device)
        comm_counts.scatter_add_(0, scatter_index.unsqueeze(1), torch.ones_like(z_norm[:, :1]))
        m_k = comm_sum / (comm_counts + 1e-6) 
        
        # Normalize community centers (m_k_norm)
        m_k_norm = F.normalize(m_k, dim=1)

        # --- 3. Calculate InfoNCE Loss ---
        
        # 3.1 Similarity matrix: training set nodes z_train vs all community centers m_k
        z_train_norm = F.normalize(z_train, dim=1)
        similarity_matrix = torch.matmul(z_train_norm, m_k_norm.transpose(0, 1)) / self.temp

        # 3.2 Construct labels (positive sample indices)
        # Find the positions of community centers corresponding to training set nodes in m_k_norm
        pos_indices = comm_id_map[comm_ids_train].to(self.device)
        
        # 3.3 Calculate InfoNCE Loss
        loss_nc = F.cross_entropy(similarity_matrix, pos_indices)
        
        return loss_nc


    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Define Benign Flow training mask (ensure only benign traffic is used for loss calculation)
        # Assume data.y=0 is Benign, and data.y!=-1 is Flow node
        # ⚠️ Note: You need to ensure that data.y reflects the training set division of Benign Flow in data.py or main()
        train_mask = (data.y == 0) # Assume data.y=0 only contains Benign Flow of training set
        
        if train_mask.sum() == 0:
            return 0.0 # Avoid empty training set
            
        # 2. Graph augmentation to generate two views
        (x1, ei1), (x2, ei2) = self.augment_graph(data)
        
        # 3. Forward propagation
        _, z1_proj = self.model(x1, ei1, data.community_id) #drop edge
        _, z2_proj = self.model(x2, ei2, data.community_id) #mask feature
        
        # 4. Calculate Loss only for Benign Flow nodes
        z1_train = z1_proj[train_mask]
        z2_train = z2_proj[train_mask]
        
        # --- L_NN (Node-Node contrastive loss) ---
        L_nn = self.info_nce_loss(z1_train, z2_train)
        
        # --- L_NC (Node-Community contrastive loss) ---
        # ⚠️ Note: Here we pass the original flow_mask and train_mask
        # flow_mask still needs to be obtained from data, but can be simplified if train_mask already ensures Flow nodes
        # Assume train_mask already ensures Flow nodes
        L_nc = self.calculate_node_community_loss(z1_proj, data.community_id, None, train_mask)
        
        # --- Total loss and backpropagation ---
        #2025-12-27 Ablation experiment, remove L_nn
        loss = self.lambda_nn * L_nn + self.lambda_nc * L_nc
        #loss = L_nn
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_epoch(self, data):
        """
        Perform gradient-free calculation on the validation set and return the total loss.
        
        Note: This method has exactly the same logic as train_epoch, but does not perform gradient calculation and backpropagation.
        """
        self.model.eval() # Switch to evaluation mode
        
        with torch.no_grad():
            # 1. Define Benign Flow validation mask
            # Assume data.y=0 is Benign Flow node (validation set)
            val_mask = (data.y == 0) 
            
            if val_mask.sum() == 0:
                return 0.0 # Avoid empty validation set

            # 2. Graph augmentation to generate two views (consistent with training)
            (x1, ei1), (x2, ei2) = self.augment_graph(data)
            
            # 3. Forward propagation
            # Directly use the original data.community_id for community IDs, consistent with train_epoch
            _, z1_proj = self.model(x1, ei1, data.community_id) 
            _, z2_proj = self.model(x2, ei2, data.community_id) 
            
            # 4. Calculate Loss only for Benign Flow nodes
            z1_val = z1_proj[val_mask]
            z2_val = z2_proj[val_mask]
            
            # --- L_NN (Node-Node contrastive loss) ---
            # Directly call self.info_nce_loss
            L_nn = self.info_nce_loss(z1_val, z2_val)
            
            # --- L_NC (Node-Community contrastive loss) ---
            # Pass the complete z1_proj to calculate all community centers, along with val_mask
            # Note: It is assumed here that val_mask already contains Flow node information
            L_nc = self.calculate_node_community_loss(z1_proj, data.community_id, None, val_mask)
            
            # --- Total loss ---
            total_loss = self.lambda_nn * L_nn + self.lambda_nc * L_nc
            #total_loss = L_nn
        return total_loss.item()    