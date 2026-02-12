import networkx as nx
import community as community_louvain
import numpy as np
from scipy.stats import pearsonr
import torch
from torch_geometric.data import Data
from scipy.stats import pearsonr
from fastdtw import fastdtw
import pywt

# Step 1: Wavelet transform to extract features (decompose frequency features of sequences)
def wavelet_extract_features2(seq):
    # Select wavelet basis (e.g., db4, suitable for non-stationary signals), decompose into 3 levels
    coeffs = pywt.wavedec(seq, 'db4', level=3)
    # Extract statistical features of wavelet coefficients (mean, variance, energy)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.var(coeff), np.sum(coeff**2)])
    return np.array(features)

def wavelet_extract_features(seq):
    coeffs = pywt.wavedec(seq, 'db4', level=3)
    features = []
    total_energy = sum([np.sum(c**2) for c in coeffs]) + 1e-6
    
    for c in coeffs:
        # 1. Relative energy ratio: Reflects the importance of this frequency band
        energy_ratio = np.sum(c**2) / total_energy
        # 2. Spectral entropy: Reflects the randomness of the signal (attack flows usually have higher entropy values)
        p = np.square(c) / (np.sum(np.square(c)) + 1e-6)
        entropy = -np.sum(p * np.log2(p + 1e-6))
        
        features.extend([np.mean(c), np.std(c), energy_ratio, entropy])
    return np.array(features)

class OptimizedGraphBuilder:
    def __init__(self, time_win=0.5, corr_th=0.85):
        self.time_win = time_win
        self.corr_th = corr_th
        self.G = nx.Graph()
    

    def build(self, processed_data):
        print(f"Building graph with {len(processed_data)} flows...")
        
        # 1. Node construction and index establishment
        # We need to index flows by 'destination IP' for quick lookup of flows entering the node
        # And index flows by 'source IP' for quick lookup of flows leaving the node
        flows_in = {}  # Key: Host IP, Value: List of flows entering Host
        flows_out = {} # Key: Host IP, Value: List of flows leaving Host
        
        flow_node_indices = {} # Map flow_file_id -> graph_node_index
        host_node_indices = {} # Map ip -> graph_node_index
        
        node_features = []
        node_types = [] # 0: Host, 1: Flow
        
        # --- Add Host nodes ---
        all_ips = set()
        for item in processed_data:
            all_ips.add(item['meta']['src'])
            all_ips.add(item['meta']['dst'])
        all_ips = sorted(all_ips)  # Fix IP traversal order    
        for ip in all_ips:
            host_node_indices[ip] = len(node_features)
            # Host features initialized to all 0 (or use one-hot encoded subnet segments)
            node_features.append(np.zeros(17, dtype=np.float32)) 
            node_types.append(0)
            self.G.add_node(host_node_indices[ip], type='host')

        # --- Add Flow nodes and communication edges ---
        for item in processed_data:
            meta = item['meta']
            idx = len(node_features)
            flow_node_indices[meta['file_id']] = idx
            
            # Add node attributes
            node_features.append(item['x'])
            node_types.append(1)
            self.G.add_node(idx, type='flow', label=meta['label'], community=-1) # Initial community is -1
            
            # Build index for subsequent search
            dst = meta['dst']
            src = meta['src']
            
            if dst not in flows_in: flows_in[dst] = []
            flows_in[dst].append(item)
            
            if src not in flows_out: flows_out[src] = []
            flows_out[src].append(item)
            
            # Add communication edges (Host <-> Flow)
            h_src_idx = host_node_indices[src] # src IP index
            h_dst_idx = host_node_indices[dst] # dst IP index
            self.G.add_edge(h_src_idx, idx, edge_type=0, weight = 1) # edge_type 0: physical
            #self.G.add_edge(idx, h_dst_idx, edge_type=0) #idx: flow
            self.G.add_edge(h_dst_idx, idx, edge_type=0, weight = 1) #idx: flow

        # --- 2. Optimized Latent Proxy Edge Inference (Sliding Window) ---
        print("Inferring latent proxy edges...")
        proxy_edges = []
        
        # Iterate over each possible intermediate node (Pivot Host)
        #common_hosts = set(flows_in.keys()) & set(flows_out.keys()) 
        common_hosts = set(flows_in.keys()) & set(flows_out.keys())
        # At this time, assume a certain IP acts as both initiator and receiver
        # What if the proxy mode is a star structure? There may only be one hop at this time, how to handle it? Some modifications need to be made here
        
        for host in common_hosts:
            # Get flows entering and leaving the host
            in_list = sorted(flows_in[host], key=lambda x: x['meta']['end_time'])
            out_list = sorted(flows_out[host], key=lambda x: x['meta']['start_time'])
            
            # Sliding window matching
            out_idx = 0
            for f_in in in_list:
                t_end_in = f_in['meta']['end_time']
                
                # Move out_idx until out_flow.start_time >= t_end_in
                # (Proxy forwarding must occur after reception)
                while out_idx < len(out_list) and out_list[out_idx]['meta']['start_time'] < t_end_in:
                    out_idx += 1
                    
                # Check subsequent flows within the window
                curr = out_idx
                while curr < len(out_list):
                    f_out = out_list[curr]
                    # Stop searching for current f_in if exceeding time window
                    if f_out['meta']['start_time'] - t_end_in > self.time_win:
                        break 
                    # Time condition is satisfied at this time, calculate feature correlation
                    # Calculate only when the lengths of the two flows are sufficient
                    seq1 = f_in['meta']['raw_lengths']
                    seq2 = f_out['meta']['raw_lengths']
                    min_len = min(len(seq1), len(seq2))

                    # Simple length truncation alignment
                    u = flow_node_indices[f_in['meta']['file_id']]
                    v = flow_node_indices[f_out['meta']['file_id']]
                    #corr, _ = pearsonr(seq1[:min_len], seq2[:min_len])
                    corr = np.exp(-(f_out['meta']['start_time'] - t_end_in)/self.time_win)
                    self.G.add_edge(u, v, edge_type=1, weight=corr) # edge_type 1: proxy
                    '''
                    if(max(len(seq1), len(seq2)) - min(len(seq1), len(seq2)) > 3):
                        curr += 1
                        continue
                    if min_len > 8:
                        feat_a = wavelet_extract_features(seq1)#wavelet transform
                        feat_b = wavelet_extract_features(seq2)#wavelet transform
                        #corr, _ = pearsonr(seq1[:min_len], seq2[:min_len])
                        corr, _ = pearsonr(feat_a, feat_b)
                        if corr > 0.95:
                            # Found latent proxy edge
                            u = flow_node_indices[f_in['meta']['file_id']]
                            v = flow_node_indices[f_out['meta']['file_id']]
                            self.G.add_edge(u, v, edge_type=1, weight=corr) # edge_type 1: proxy 
                    '''
                    curr += 1

        
        common_hosts = set(flows_out.keys()) | set(flows_in.keys())
        #common_hosts = set(flows_out.keys())
        # At this time, assume a certain IP acts as both initiator and receiver
        # What if the proxy mode is a star structure? There may only be one hop at this time, how to handle it? Some modifications need to be made here
        
        for host in common_hosts:
            if(host == '172.16.0.1'):
                a = 5
            # 1. 🚨 Key modification: Merge all incoming and outgoing flows, no longer distinguish directions
            all_flows = flows_in.get(host, []) + flows_out.get(host, [])
            #all_flows = flows_out.get(host, [])
            # 2. Sort all flows by start time
            all_flows.sort(key=lambda x: x['meta']['start_time'])
            
            N = len(all_flows)
            
            # 3. Double-layer loop to implement acausal sliding window matching
            for i in range(N):
                f_a = all_flows[i]
                t_start_a = f_a['meta']['start_time']
                
                # Start from the next flow of i (avoid duplicate matching and self-loops)
                for j in range(i + 1, N):
                    f_b = all_flows[j]
                    t_start_b = f_b['meta']['start_time']
                    
                    # Since the list is sorted by t_start, t_start_b >= t_start_a
                    time_diff = t_start_b - t_start_a
                    
                    # If the time difference exceeds the window, subsequent f_b' will also exceed the window due to sorted list, break inner loop directly
                    #if time_diff > self.time_win:
                    if time_diff > 10:
                        break
                        
                    # Time condition is satisfied at this time: time_diff <= self.time_win
                    
                    # 4. Calculate feature correlation
                    seq1 = f_a['meta']['raw_lengths']
                    seq2 = f_b['meta']['raw_lengths']
                    
                    # Simple length truncation alignment
                    min_len = min(len(seq1), len(seq2))
                    if(max(len(seq1), len(seq2)) - min(len(seq1), len(seq2))) > 10:#Cannot be set to 1 because packet loss is likely due to network reasons
                        continue
                    
                    if min_len > 8:
                        # Assume pearsonr is imported
                        feat_a = wavelet_extract_features(seq1)#wavelet transform
                        feat_b = wavelet_extract_features(seq2)#wavelet transform
                        corr, _ = pearsonr(feat_a, feat_b)
                        #corr, _ = pearsonr(seq1[:min_len], seq2[:min_len])
                        
                        #if corr > self.corr_th:
                        if corr > 0.95:
                            # Found latent similarity/co-occurrence edges (acausal)
                            u = flow_node_indices[f_a['meta']['file_id']]
                            v = flow_node_indices[f_b['meta']['file_id']]
                            
                            # Establish bidirectional edges because there is no longer a strict causal relationship
                            self.G.add_edge(u, v, edge_type=1, weight=corr) # edge_type 1: Similarity
                            #self.G.add_edge(v, u, edge_type=1, weight=corr) # edge_type 1: Similarity
        
        # --- 3. Community Detection (Louvain) ---
        print("Detecting communities...")
        # Partition only based on unweighted graph or communication edges to avoid being misled by malicious proxy edges
        # Or use a weighted graph, but need to adjust weights carefully
        #G_undirected = self.G.to_undirected()
        
        #partition = community_louvain.best_partition(G_undirected)
        partition = community_louvain.best_partition(self.G,random_state=42, weight = 'weight')
        
        # Update community ID to node features (use Embedding during actual training)
        community_ids = []
        for i in range(len(node_features)):
            cid = partition.get(i, 0)
            community_ids.append(cid)
            self.G.nodes[i]['community'] = cid

        # --- 4. Convert to PyG Data ---
        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        y = torch.tensor([self.G.nodes[i].get('label', -1) for i in range(len(node_features))], dtype=torch.long)
        c_ids = torch.tensor(community_ids, dtype=torch.long)
        
        # Edge type attributes (used for Relational GCN or Edge weighting)
        edge_attr = []
        for u, v in self.G.edges:
            edge_attr.append(self.G[u][v].get('edge_type', 0))
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, community_id=c_ids, edge_attr=edge_attr)