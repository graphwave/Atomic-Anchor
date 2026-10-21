import networkx as nx
import community as community_louvain
import numpy as np
import random
from scipy.stats import pearsonr
import torch
from torch_geometric.data import Data
from scipy.stats import pearsonr
from fastdtw import fastdtw
import pywt
import time

def wavelet_extract_features2(seq):

    coeffs = pywt.wavedec(seq, 'db4', level=3)

    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.var(coeff), np.sum(coeff**2)])
    return np.array(features)

def wavelet_extract_features(seq):
    coeffs = pywt.wavedec(seq, 'db4', level=3)
    features = []
    total_energy = sum([np.sum(c**2) for c in coeffs]) + 1e-6

    for c in coeffs:

        energy_ratio = np.sum(c**2) / total_energy

        p = np.square(c) / (np.sum(np.square(c)) + 1e-6)
        entropy = -np.sum(p * np.log2(p + 1e-6))

        features.extend([np.mean(c), np.std(c), energy_ratio, entropy])
    return np.array(features)

def _ap_local_moving(adj, hub, internal, gamma_sem, gamma_phy, tau,
                     min_gain, max_passes, rng):
    node_comm = {u: u for u in adj}
    sin = {0: {}, 1: {}}
    sig = {0: {}, 1: {}}
    kdeg = {0: {}, 1: {}}
    m = [0.0, 0.0]
    for u, nbrs in adj.items():
        dp = internal[u][0]
        ds = internal[u][1]
        for w2 in nbrs.values():
            dp += w2[0]
            ds += w2[1]
        kdeg[0][u] = dp
        kdeg[1][u] = ds
        sin[0][u] = internal[u][0]
        sin[1][u] = internal[u][1]
        sig[0][u] = dp
        sig[1][u] = ds
        m[0] += 0.5 * internal[u][0]
        m[1] += 0.5 * internal[u][1]
    for u, nbrs in adj.items():
        for v, w2 in nbrs.items():
            if u < v:
                m[0] += w2[0]
                m[1] += w2[1]

    comm_hub = dict(hub)
    gammas = ((0, gamma_phy), (1, gamma_sem))

    for _pass in range(max_passes):
        moved = 0
        order = [u for u in adj if not hub[u]]
        rng.shuffle(order)
        for u in order:
            cu = node_comm[u]

            w_old = [0.0, 0.0]
            nw = {}
            for v, w2 in adj[u].items():
                cv = node_comm[v]
                if cv == cu:
                    w_old[0] += w2[0]
                    w_old[1] += w2[1]
                elif not comm_hub[cv]:
                    e = nw.get(cv)
                    if e is None:
                        nw[cv] = [w2[0], w2[1]]
                    else:
                        e[0] += w2[0]
                        e[1] += w2[1]

            for t in (0, 1):
                sig[t][cu] -= kdeg[t][u]
                sin[t][cu] -= 2.0 * w_old[t]

            ks = kdeg[1][u]
            best_c = None
            best_dq = min_gain
            for c, w2 in nw.items():

                if tau > 0.0 and m[1] > 0.0 and ks > 0.0 and (w2[1] / ks) < tau:
                    continue
                dq = 0.0
                for t, g in gammas:
                    if m[t] <= 0.0 or g == 0.0:
                        continue
                    a = 1.0 / (2.0 * m[t])
                    dq += g * (2.0 * a * w2[t]
                               - 2.0 * a * a * sig[t][c] * kdeg[t][u])
                if dq > best_dq:
                    best_dq = dq
                    best_c = c

            if best_c is not None:
                node_comm[u] = best_c
                for t in (0, 1):
                    sin[t][best_c] += 2.0 * nw[best_c][t]
                    sig[t][best_c] += kdeg[t][u]
                moved += 1
            else:

                for t in (0, 1):
                    sig[t][cu] += kdeg[t][u]
                    sin[t][cu] += 2.0 * w_old[t]
        if moved == 0:
            break

    Q = 0.0
    active = set(node_comm.values())
    for t, g in gammas:
        if m[t] <= 0.0 or g == 0.0:
            continue
        a = 1.0 / (2.0 * m[t])
        for c in active:
            Q += g * (sin[t][c] * a - (sig[t][c] * a) ** 2)
    return node_comm, sin, m, Q

def atomicity_preserving_louvain(G, gamma_sem=1.0, gamma_phy=0.0, tau=0.0,
                                 attach_hosts=True, min_comm_size=3,
                                 min_gain=1e-9,
                                 max_passes=200, max_level=32, random_state=42):
    rng = random.Random(random_state)

    adj = {n: {} for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get('weight', 1.0))
        wp, ws = (w, 0.0) if d.get('edge_type', 0) == 0 else (0.0, w)
        e = adj[u].get(v)
        if e is None:
            e = [wp, ws]
            adj[u][v] = e
            adj[v][u] = e
        else:
            e[0] += wp
            e[1] += ws

    hub = {n: (G.nodes[n].get('type') == 'host') for n in G.nodes()}
    internal = {u: [0.0, 0.0] for u in adj}
    mapping = {u: u for u in adj}

    cur_adj, cur_hub, cur_internal = adj, hub, internal
    for _level in range(max_level):
        node_comm, sin, m, Q = _ap_local_moving(
            cur_adj, cur_hub, cur_internal,
            gamma_sem, gamma_phy, tau, min_gain, max_passes, rng)

        comms = sorted(set(node_comm.values()))
        if len(comms) == len(cur_adj):
            break
        remap = {c: i for i, c in enumerate(comms)}
        new_internal = {remap[c]: [sin[0][c], sin[1][c]] for c in comms}
        new_hub = {remap[c]: False for c in comms}
        for u in cur_adj:
            if cur_hub[u]:
                new_hub[remap[node_comm[u]]] = True
        new_adj = {i: {} for i in range(len(comms))}
        for u, nbrs in cur_adj.items():
            cu = remap[node_comm[u]]
            for v, w2 in nbrs.items():
                if u < v:
                    cv = remap[node_comm[v]]
                    if cu != cv:
                        e = new_adj[cu].get(cv)
                        if e is None:
                            new_adj[cu][cv] = [w2[0], w2[1]]
                            new_adj[cv][cu] = new_adj[cu][cv]
                        else:
                            e[0] += w2[0]
                            e[1] += w2[1]
        for u in mapping:
            mapping[u] = remap[node_comm[mapping[u]]]
        cur_adj, cur_hub, cur_internal = new_adj, new_hub, new_internal

    flow_sups = sorted([s for s in cur_adj if not cur_hub[s]])
    cid = {s: i for i, s in enumerate(flow_sups)}
    next_id = len(flow_sups)
    for s in cur_adj:
        if not cur_hub[s]:
            continue
        if attach_hosts:
            best_v, best_w = None, -1.0
            for v, w2 in cur_adj[s].items():
                tw = w2[0] + w2[1]
                if tw > best_w:
                    best_w, best_v = tw, v
        else:
            best_v = None
        if best_v is not None:
            cid[s] = cid[best_v]
        else:
            cid[s] = next_id
            next_id += 1

    if min_comm_size > 1:
        node_cid = {u: cid[mapping[u]] for u in mapping}
        for _round in range(max_level):
            members = {}
            for u, c in node_cid.items():
                if G.nodes[u].get('type') == 'host':
                    continue
                members.setdefault(c, []).append(u)
            small = [c for c, ms in members.items() if len(ms) < min_comm_size]
            if not small:
                break
            changed = False
            for c in small:

                score = {}
                for u in members[c]:
                    for v, d in G[u].items():
                        tc = node_cid.get(v)
                        if tc is None or tc == c:
                            continue
                        score[tc] = score.get(tc, 0.0) + float(d.get('weight', 1.0))
                if not score:
                    continue
                target = max(score.items(), key=lambda kv: kv[1])[0]
                for u in members[c]:
                    node_cid[u] = target
                changed = True
            if not changed:
                break

        uniq = sorted(set(node_cid.values()))
        rid = {c: i for i, c in enumerate(uniq)}
        return {u: int(rid[node_cid[u]]) for u in mapping}

    return {u: int(cid[mapping[u]]) for u in mapping}

def semantic_refine_ids(cid, X, y, ei, ea, min_absorb=3):
    from collections import defaultdict
    refined = cid.copy()
    adj = defaultdict(list)
    for (u, v), t in zip(ei.T, ea):
        adj[int(u)].append((int(v), int(t)))
        adj[int(v)].append((int(u), int(t)))
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        flows = idx[y[idx] >= 0]
        fset = set(flows.tolist())
        sem = [(u, v) for u in flows for v, t in adj[u]
               if t == 1 and v in fset and u < v]
        if not sem:
            continue
        GS = nx.Graph(); GS.add_nodes_from(flows.tolist()); GS.add_edges_from(sem)
        comps = sorted([sorted(comp) for comp in nx.connected_components(GS)
                        if len(comp) >= min_absorb], key=len, reverse=True)
        if len(comps) < 2:
            continue
        comps = [np.array(comp) for comp in comps]
        cents = np.stack([X[comp].mean(axis=0) for comp in comps])
        for i, comp in enumerate(comps):
            refined[comp] = 10**6 + int(c) * 1000 + i
        rest = np.array([n for n in flows if refined[n] == c])
        if len(rest):
            d = np.linalg.norm(X[rest][:, None, :] - cents[None, :, :], axis=2)
            for j, n in enumerate(rest):
                refined[n] = 10**6 + int(c) * 1000 + int(d[j].argmin())
    uniq = {v: i for i, v in enumerate(sorted(set(refined.tolist())))}
    return np.array([uniq[v] for v in refined.tolist()], dtype=np.int64)

class OptimizedGraphBuilder:
    def __init__(self, causal_win=0.5, similarity_win=30, corr_th=0.8,
                 partition='ap', min_comm_size=3, ap_tau=0.0, ap_gphy=0.0):
        self.causal_win = causal_win
        self.similarity_win = similarity_win
        self.corr_th = corr_th
        self.partition = partition
        self.min_comm_size = min_comm_size
        self.ap_tau = ap_tau
        self.ap_gphy = ap_gphy
        self.G = nx.Graph()

    @staticmethod
    def _refine_communities(G, partition, node_features, min_absorb=3,
                            max_frac=0.5, min_dist=1.0):
        import numpy as _np
        X = _np.asarray(node_features)
        refined = {}
        comm_flows = {}
        for n in G.nodes():
            if G.nodes[n].get('type') == 'flow':
                comm_flows.setdefault(partition[n], []).append(n)
                refined[n] = None
            else:
                refined[n] = (partition[n], -1)
        n_split = 0
        for c, flows in comm_flows.items():
            fset = set(flows)
            sem = [(u, v) for u in flows for v in sorted(set(G[u]) & fset)
                   if G[u][v].get('edge_type', 0) == 1 and u < v]
            if not sem:
                continue
            GS = nx.Graph(); GS.add_nodes_from(flows); GS.add_edges_from(sem)
            comps = sorted([sorted(comp) for comp in nx.connected_components(GS)
                            if len(comp) >= min_absorb], key=len, reverse=True)
            if len(comps) < 2:
                continue
            sub_i = 0
            for comp in comps:
                if len(comp) > max_frac * len(flows):
                    continue
                cent = X[comp].mean(axis=0)
                others = [n for n in flows if n not in set(comp)]
                if not others:
                    continue
                d = _np.linalg.norm(cent - X[others].mean(axis=0))
                if d < min_dist:
                    continue
                for n in comp:
                    refined[n] = (c, sub_i + 1000)
                sub_i += 1
                n_split += 1
        print(f"Stage-2 surgical refinement: carved {n_split} semantic subgroups")

        for n, v in refined.items():
            if v is None:
                refined[n] = (partition[n], -1)

        uniq = {c: i for i, c in enumerate(sorted(set(refined.values())))}
        return {n: uniq[refined[n]] for n in G.nodes()}

    def build(self, processed_data, pre_louvain_hook=None):
        print(f"Building graph with {len(processed_data)} flows...")

        flows_in = {}
        flows_out = {}

        flow_node_indices = {}
        host_node_indices = {}

        node_features = []
        node_types = []

        all_ips = set()
        for item in processed_data:
            all_ips.add(item['meta']['src'])
            all_ips.add(item['meta']['dst'])
        all_ips = sorted(all_ips)
        for ip in all_ips:
            host_node_indices[ip] = len(node_features)

            node_features.append(np.zeros(17, dtype=np.float32)) 
            node_types.append(0)
            self.G.add_node(host_node_indices[ip], type='host')

        for item in processed_data:
            meta = item['meta']
            idx = len(node_features)
            flow_node_indices[meta['file_id']] = idx

            node_features.append(item['x'])
            node_types.append(1)
            self.G.add_node(idx, type='flow', label=meta['label'], community=-1)

            dst = meta['dst']
            src = meta['src']

            if dst not in flows_in: flows_in[dst] = []
            flows_in[dst].append(item)

            if src not in flows_out: flows_out[src] = []
            flows_out[src].append(item)

            h_src_idx = host_node_indices[src]
            h_dst_idx = host_node_indices[dst]
            self.G.add_edge(h_src_idx, idx, edge_type=0, weight = 1)

            self.G.add_edge(h_dst_idx, idx, edge_type=0, weight = 1)

        print("Inferring latent proxy edges...")
        proxy_edges = []

        if self.causal_win <= 0:
            common_hosts = []
        common_hosts = sorted(set(flows_in.keys()) & set(flows_out.keys()))

        for host in common_hosts:

            in_list = sorted(flows_in[host], key=lambda x: x['meta']['end_time'])
            out_list = sorted(flows_out[host], key=lambda x: x['meta']['start_time'])

            out_idx = 0
            for f_in in in_list:
                t_end_in = f_in['meta']['end_time']

                while out_idx < len(out_list) and out_list[out_idx]['meta']['start_time'] < t_end_in:
                    out_idx += 1

                curr = out_idx
                while curr < len(out_list):
                    f_out = out_list[curr]

                    if f_out['meta']['start_time'] - t_end_in > self.causal_win:
                        break 

                    seq1 = f_in['meta']['raw_lengths']
                    seq2 = f_out['meta']['raw_lengths']
                    min_len = min(len(seq1), len(seq2))

                    u = flow_node_indices[f_in['meta']['file_id']]
                    v = flow_node_indices[f_out['meta']['file_id']]

                    corr = np.exp(-(f_out['meta']['start_time'] - t_end_in)/self.causal_win)
                    self.G.add_edge(u, v, edge_type=1, weight=corr)
                    curr += 1

        common_hosts = sorted(set(flows_out.keys()) | set(flows_in.keys()))

        for host in common_hosts:
            if(host == '172.16.0.1'):
                a = 5

            all_flows = flows_in.get(host, []) + flows_out.get(host, [])

            all_flows.sort(key=lambda x: x['meta']['start_time'])

            N = len(all_flows)

            if self.similarity_win <= 0:
                N = 0

            for i in range(N):
                f_a = all_flows[i]
                t_start_a = f_a['meta']['start_time']
                count = 0

                for j in range(i + 1, N):
                    f_b = all_flows[j]
                    t_start_b = f_b['meta']['start_time']

                    time_diff = t_start_b - t_start_a

                    if time_diff > self.similarity_win:
                        break

                    seq1 = f_a['meta']['raw_lengths']
                    seq2 = f_b['meta']['raw_lengths']

                    min_len = min(len(seq1), len(seq2))
                    if(max(len(seq1), len(seq2)) - min(len(seq1), len(seq2))) > 10:
                        continue

                    if min_len > 3:

                        feat_a = wavelet_extract_features(seq1)
                        feat_b = wavelet_extract_features(seq2)
                        corr, _ = pearsonr(feat_a, feat_b)

                        if corr > self.corr_th:

                            u = flow_node_indices[f_a['meta']['file_id']]
                            v = flow_node_indices[f_b['meta']['file_id']]

                            self.G.add_edge(u, v, edge_type=1, weight=corr)
                            count += 1
                            if(count >= 1):
                                break

        if pre_louvain_hook is not None:
            pre_louvain_hook(self.G, flow_node_indices, host_node_indices)

        print("Detecting communities...")

        start_time = time.time()
        if self.partition == 'ap':

            partition = atomicity_preserving_louvain(
                self.G, gamma_phy=self.ap_gphy, tau=self.ap_tau,
                random_state=42, min_comm_size=self.min_comm_size)
        elif self.partition == 'refined':

            coarse_partition = community_louvain.best_partition(
                self.G, random_state=42, weight='weight')
            self.partition_coarse = dict(coarse_partition)
            partition = self._refine_communities(self.G, coarse_partition, node_features)
            print(f"Stage-2 refinement: {len(set(partition.values()))} sub-communities")
        elif self.partition == 'hier':

            partition = community_louvain.best_partition(
                self.G, random_state=42, weight='weight')
            self.fine_partition = self._refine_communities(
                self.G, partition, node_features)
            print(f"Stage-2 surgical refinement: {len(set(partition.values()))} coarse communities, "
                  f"{len(set(self.fine_partition.values()))} fine sub-communities")
        else:

            partition = community_louvain.best_partition(
                self.G, random_state=42, weight='weight')
            self.partition_coarse = dict(partition)
        end_time = time.time()
        print(f"Community partitioning time: {end_time-start_time}s")

        community_ids = []
        for i in range(len(node_features)):
            cid = partition.get(i, 0)
            community_ids.append(cid)
            self.G.nodes[i]['community'] = cid

        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        y = torch.tensor([self.G.nodes[i].get('label', -1) for i in range(len(node_features))], dtype=torch.long)
        c_ids = torch.tensor(community_ids, dtype=torch.long)

        edge_attr = []
        for u, v in self.G.edges:
            edge_attr.append(self.G[u][v].get('edge_type', 0))
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        if self.partition == 'hier':
            fine_ids = [self.fine_partition[i] for i in range(len(node_features))]
            c_fine = torch.tensor(fine_ids, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y, community_id=c_ids,
                        community_id_fine=c_fine, edge_attr=edge_attr)
        data = Data(x=x, edge_index=edge_index, y=y, community_id=c_ids, edge_attr=edge_attr)

        coarse_map = getattr(self, 'partition_coarse', None)
        if coarse_map is not None:
            coarse_ids = [coarse_map.get(i, 0) for i in range(len(node_features))]
            data.community_id_coarse = torch.tensor(coarse_ids, dtype=torch.long)
        return data

