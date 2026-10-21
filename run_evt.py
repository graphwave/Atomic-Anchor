import torch
import numpy as np
from sklearn.svm import OneClassSVM
from scipy.stats import genpareto
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
import random

from feature_extract import AdvancedFeatureExtractor, load_or_extract_features
from graph_builder import OptimizedGraphBuilder

from hcg_model import CommunityGuidedGAT
from train import HCGTrainer
import pandas as pd
import copy
import pickle
import os
import argparse
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_evt_threshold(train_scores, initial_quantile=0.95, risk_prob=1e-2):

    t = np.quantile(train_scores, initial_quantile)

    excesses = train_scores[train_scores > t] - t

    if len(excesses) < 10:
        print("Warning: Too few samples in the tail for EVT fitting. Falling back to quantile.")
        return np.quantile(train_scores, 1 - risk_prob)

    shape, loc, scale = genpareto.fit(excesses)

    n = len(train_scores)
    n_t = len(excesses)

    if abs(shape) < 1e-4:
        evt_t = t + scale * np.log(n_t / (n * risk_prob))
    else:
        evt_t = t + (scale / shape) * (np.power(n_t / (n * risk_prob), shape) - 1)

    return evt_t

def normalize_features(data_list):
    for d in data_list:

        features = d['x']

        features[np.isnan(features)] = 0.0

        features[np.isinf(features)] = 0.0

        d['x'] = features

    return data_list

def z_score_standardize(train_data_list, test_data_list):
    print("--- Starting Z-Score Feature Standardization...")

    train_features = np.array([d['x'] for d in train_data_list])

    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)

    std[std < 1e-6] = 1.0 

    for d in train_data_list:
        d['x'] = (d['x'] - mean) / std

    for d in test_data_list:
        d['x'] = (d['x'] - mean) / std

    print("Feature Standardization completed.")
    return train_data_list, test_data_list

def loo_min_distances(distances, embeddings, comm_ids, centroids):
    comm_list = list(centroids.keys())
    col_of = {c: j for j, c in enumerate(comm_list)}
    unique_ids = np.unique(comm_ids)
    comm_sizes = {c: int(np.sum(comm_ids == c)) for c in unique_ids}
    out = np.empty(len(embeddings), dtype=np.float64)
    for i in range(len(embeddings)):
        ci = comm_ids[i]
        n_k = comm_sizes[ci]
        row = distances[i].copy()
        if n_k > 1:
            mu_loo = (centroids[ci] * n_k - embeddings[i]) / (n_k - 1)
            row[col_of[ci]] = np.linalg.norm(embeddings[i] - mu_loo)
        else:
            row[col_of[ci]] = np.inf
        out[i] = row.min()
    return out

def read_pkl_file(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"Warning: PKL file {pkl_path} not found, skipping")
        return []
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if 'ids2017' in pkl_path:
            valid_data = [item for item in data if item is not None and item['x'][13] > 3 and item['x'][15] < 50\
                        and item['x'][16] >0]
            print(f"loaded {pkl_path}: {len(valid_data)} valid records")
            return valid_data
        else:
            return data
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}, skipping")
        return []

def main():

    parser = argparse.ArgumentParser(description='Run model with specified time_win parameter')
    parser.add_argument('--causal_win', type=float, required=False, default = 0.1, help='Time window parameter for causal connection')
    parser.add_argument('--similarity_win', type=float, required=False, default = 30, help='Time window parameter for similarity')
    parser.add_argument('--corr_th', type=float, required=False, default = 0.95, help='similarity level for DWT')
    parser.add_argument('--partition', type=str, required=False, default='ap', choices=['ap', 'standard', 'refined', 'hier'],
                        help='community partition method: ap (AP-Louvain), standard (Louvain), refined (full semantic refinement), hier (coarse CCI + fine IIM)')
    parser.add_argument('--tag_suffix', type=str, required=False, default='',
                        help='suffix appended to part_tag (isolates sensitivity-experiment outputs, empty by default)')
    parser.add_argument('--min_comm', type=int, required=False, default=3,
                        help='AP-Louvain min community size (anchor support floor)')
    parser.add_argument('--ap_tau', type=float, required=False, default=0.0,
                        help='AP-Louvain atomicity admission threshold tau (0=off, ablation {0,0.2,0.4})')
    parser.add_argument('--ap_gphy', type=float, required=False, default=0.0,
                        help='AP-Louvain physical-edge modularity weight gamma_phy (0=off)')
    parser.add_argument('--dump_only', action='store_true',
                        help='build graphs + dump communities/features, then exit (no training)')
    parser.add_argument('--save_features', action='store_true',
                        help='save per-split graph features npz along with community dumps')
    parser.add_argument('--no_filter', action='store_true',
                        help='skip dataset/attack filtering (raw unfiltered data; result dir gets a _nofilt suffix)')
    parser.add_argument('--ablate', choices=['none', 'cci', 'iim', 'lnn', 'lnc'], default='none',
                        help='module ablation: cci=remove contextual injection, iim=remove isolation masking, lnn=remove L_NN, lnc=remove L_NC (result dir gets an _ablateXXX suffix)')
    parser.add_argument('--sgv', action='store_true',
                        help='SGV semantic-group verification: adds a subgroup-level second hypothesis test on the std pipeline (OR fusion); main channel unchanged')
    parser.add_argument('--sgv_c', type=float, required=False, default=2.0,
                        help='SGV safety factor C: subgroup verdict threshold = C * (1-q)-quantile of val subgroup distances '
                             '(attack subgroups exceed the null tail by 3.8-237x, benign drift <=1.61x; C=2 in between)')
    parser.add_argument('--sgv_q', type=float, required=False, default=0.001,
                        help='tail quantile q of the SGV val null distribution (threshold taken at the (1-q)-quantile)')
    parser.add_argument('--sgv_min_k', type=int, required=False, default=3,
                        help='minimum SGV subgroup size (subgroups below it are excluded from testing)')
    parser.add_argument('--pe', type=float, required=False, default=0.3,
                        help='View1 edge-dropping probability pe (contrastive augmentation)')
    parser.add_argument('--pf', type=float, required=False, default=0.3,
                        help='View2 feature-masking probability pf (contrastive augmentation)')
    parser.add_argument('--max_epochs', type=int, required=False, default=1000,
                        help='max training epochs')
    parser.add_argument('--patience', type=int, required=False, default=800,
                        help='early-stopping patience (val loss)')
    parser.add_argument('--attack', type=str, required=False, default='Zeus.pkl',
                        help='attack pkl file name under data root (e.g. Neris.pkl)')
    parser.add_argument('--ensemble', action='store_true',
                        help='dual-granularity ensemble: score fusion of the coarse topological view and the fine semantic view')
    parser.add_argument('--threshold_mode', type=str, required=False, default='evt',
                        choices=['evt', 'quantile', 'comm_evt', 'comm_evt_val'],
                        help='threshold calibration: evt (global EVT/POT), quantile (fixed Q99.5), '
                             'comm_evt (per-nearest-centroid-community EVT, train-LOO source), '
                             'comm_evt_val (per-community EVT, val source)')
    parser.add_argument('--calib_on', type=str, required=False, default='train',
                        choices=['train', 'val'],
                        help='EVT calibration source: train (training LOO distances) or val (validation distances, no LOO bias)')
    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='random seed (42 by default; use --seed 43/44 for ensemble member models)')
    parser.add_argument('--dataset', type=str, required=False, default='botnet2014',
                        choices=['botnet2014', 'ids2017', 'unsw', 'ustc'],
                        help='dataset name (selects the data/ subdirectory and the preprocessing filter)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    extractor = AdvancedFeatureExtractor()
    dataset = args.dataset
    data_root = f'data/{dataset}'

    part_tag = args.partition if args.partition == 'standard' else f'{args.partition}_mc{args.min_comm}'
    if args.tag_suffix:
        part_tag += args.tag_suffix
    part_tag += f'_e{args.max_epochs}'
    if args.ensemble:
        part_tag += '_ens'
    if args.seed != 42:
        part_tag += f'_s{args.seed}'
    if args.sgv:
        part_tag += '_sgv'
    if args.no_filter:
        part_tag += '_nofilt'
    if args.ablate != 'none':
        part_tag += f'_ablate{args.ablate.upper()}'
    result_root = 'result/result_' + dataset + '_delta=' + str(args.corr_th) + '_' + part_tag
    os.makedirs(result_root, exist_ok=True)
    MODEL_PATH = 'model/model_' + dataset + '_delta=' + str(args.corr_th) + '_' + part_tag + '_' + args.attack.replace('.pkl', '') + '.pth'
    processed_data = []

    benign_name = 'benign_sessions.pkl' if dataset == 'unsw' else 'benign.pkl'
    benign_pkl_path = os.path.join(data_root, benign_name)
    attack_pkl_name = args.attack
    attack_pkl_path = os.path.join(data_root, attack_pkl_name)

    benign_data = read_pkl_file(benign_pkl_path)
    attack_data = read_pkl_file(attack_pkl_path)

    processed_data.extend(benign_data)
    processed_data.extend(attack_data)

    benign_flows = [d for d in processed_data if d['meta']['label'] == 0]
    attack_flows = [d for d in processed_data if d['meta']['label'] == 1]

    MAX_TRAIN_FLOWS = 15000
    MAX_TEST_FLOWS = 10000

    full_size = len(benign_flows)
    random.shuffle(benign_flows)

    train_ratio = 0.6
    val_ratio = 0.2

    train_size_full = int(full_size * train_ratio)
    actual_train_size = min(train_size_full, MAX_TRAIN_FLOWS) 

    val_size = int(full_size * val_ratio)
    actual_val_size = min(val_size, MAX_TRAIN_FLOWS)

    test_size = int(full_size * (1-train_ratio-val_ratio))
    actual_test_size = min(MAX_TEST_FLOWS, test_size)

    train_flows = benign_flows[:actual_train_size]
    val_flows = benign_flows[actual_train_size: actual_train_size + actual_val_size]
    test_benign_flows = benign_flows[actual_train_size + actual_val_size : actual_train_size + actual_val_size + actual_test_size]

    test_flows = test_benign_flows + attack_flows[:min(len(attack_flows), MAX_TRAIN_FLOWS)]

    train_flows = normalize_features(train_flows)
    val_flows = normalize_features(val_flows)
    test_flows = normalize_features(test_flows)

    all_flows_for_std = train_flows + val_flows

    train_flows_std, val_flows_std = z_score_standardize(copy.deepcopy(train_flows), val_flows)
    train_flows_std, test_flows_std = z_score_standardize(copy.deepcopy(train_flows), test_flows)

    train_flows = train_flows_std
    val_flows = val_flows_std
    test_flows = test_flows_std

    builder_train = OptimizedGraphBuilder(causal_win=args.causal_win, similarity_win = args.similarity_win, corr_th=args.corr_th,
                                          partition=args.partition, min_comm_size=args.min_comm,
                                          ap_tau=args.ap_tau, ap_gphy=args.ap_gphy)
    pyg_data_train = builder_train.build(train_flows)

    builder_val = OptimizedGraphBuilder(causal_win=args.causal_win, similarity_win = args.similarity_win, corr_th=args.corr_th,
                                        partition=args.partition, min_comm_size=args.min_comm,
                                          ap_tau=args.ap_tau, ap_gphy=args.ap_gphy)
    pyg_data_val = builder_val.build(val_flows)

    start_time = time.time()
    builder_test = OptimizedGraphBuilder(causal_win=args.causal_win, similarity_win = args.similarity_win, corr_th=args.corr_th,
                                         partition=args.partition, min_comm_size=args.min_comm,
                                          ap_tau=args.ap_tau, ap_gphy=args.ap_gphy)
    pyg_data_test = builder_test.build(test_flows)
    end_time = time.time()
    print(f"Training-graph construction time: {end_time-start_time}s" )

    try:
        import pandas as _pd
        for _split, _pyg in [('train', pyg_data_train), ('val', pyg_data_val), ('test', pyg_data_test)]:
            _cid = _pyg.community_id.cpu().numpy()
            _cco = (_pyg.community_id_coarse.cpu().numpy()
                    if hasattr(_pyg, 'community_id_coarse') else _cid)
            _y = _pyg.y.cpu().numpy()
            _pd.DataFrame({'node_index': np.arange(len(_cid)), 'comm_fine': _cid,
                           'comm_coarse': _cco, 'label': _y}).to_csv(
                f"{result_root}/{attack_pkl_name}_{_split}_communities.csv", index=False)
            if getattr(args, 'save_features', False):
                np.savez_compressed(
                    f"{result_root}/{attack_pkl_name}_{_split}_graph_features.npz",
                    x=_pyg.x.cpu().numpy(), y=_y, comm_fine=_cid, comm_coarse=_cco)
        if getattr(args, 'dump_only', False):
            print("[dump-only] graph built and exported, exiting early as requested (no training)")
            return
        print(f"[dump] community assignments exported to {result_root}/{attack_pkl_name}_*_communities.csv")
    except Exception as _e:
        print(f"[warn] failed to export community assignments (main flow unaffected): {_e}")

    result_path = "./experiment_results"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    torch.save(pyg_data_test, os.path.join(result_path, dataset + "_pyg_data_test.pt"))
    print(f"Graph data saved to {result_path}/pyg_data_test.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pyg_data_train = pyg_data_train.to(device)
    pyg_data_val = pyg_data_val.to(device)
    pyg_data_test = pyg_data_test.to(device)

    num_communities_train = pyg_data_train.community_id.max().item() + 1
    FEATURE_DIM = 17 

    _ablate = args.ablate if args.ablate != 'none' else None
    model = CommunityGuidedGAT(
        feature_dim=FEATURE_DIM,
        num_communities=num_communities_train,
        hidden_dim=64,
        embed_dim=32,
        ablate=_ablate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    trainer = HCGTrainer(model, optimizer, ablate=_ablate, pe=args.pe, pf=args.pf)

    print("\n--- Step 4: Starting Contrastive Training with Validation ---")
    PATIENCE = args.patience
    MAX_EPOCHS = args.max_epochs
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None 

    ALWAYS_RETRAIN = True
    if (not ALWAYS_RETRAIN) and os.path.exists(MODEL_PATH):
        print(f"existing model file found: {MODEL_PATH}, skipping training and loading directly...")

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)

        need_training = False
    else:
        print("no reusable pre-trained model found (or retraining forced), starting training...")
        need_training = True
    if need_training:
        for epoch in range(MAX_EPOCHS):

            train_loss = trainer.train_epoch(pyg_data_train) 

            val_loss = trainer.evaluate_epoch(pyg_data_val)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_model_state = copy.deepcopy(model.state_dict())

                if epoch % 50 != 0:
                    print(f"--> Epoch {epoch}: Model saved with best VAL loss: {best_val_loss:.4f}")
                torch.save(best_model_state, MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs. Best VAL loss: {best_val_loss:.4f}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
            print("Loaded best model weights (based on validation loss) for final evaluation.")
        else:
            print("Warning: Did not find a better loss than initial, using final model state.")

    modelB = None
    fine_data = {}
    if args.ensemble:
        from graph_builder import semantic_refine_ids
        print("\n--- Step 4.5: Training fine-view model (semantic refinement) ---")
        def _build_fine(d, tag):
            fine_np = semantic_refine_ids(
                d.community_id.cpu().numpy(), d.x.cpu().numpy(),
                d.y.cpu().numpy(), d.edge_index.cpu().numpy(),
                d.edge_attr.cpu().numpy())
            from torch_geometric.data import Data as _Data
            fine_data[tag] = fine_np
            return _Data(x=d.x, edge_index=d.edge_index, y=d.y,
                         community_id=torch.tensor(fine_np, dtype=torch.long).to(d.x.device),
                         edge_attr=d.edge_attr)
        pyg_data_train_f = _build_fine(pyg_data_train, 'train')
        pyg_data_val_f = _build_fine(pyg_data_val, 'val')
        pyg_data_test_f = _build_fine(pyg_data_test, 'test')
        print(f"fine-view community counts (train/val/test) = "
              f"{pyg_data_train_f.community_id.max().item()+1}/"
              f"{pyg_data_val_f.community_id.max().item()+1}/"
              f"{pyg_data_test_f.community_id.max().item()+1}")

        MODEL_PATH_FINE = MODEL_PATH.replace('.pth', '_fine.pth')
        modelB = CommunityGuidedGAT(
            feature_dim=FEATURE_DIM,
            num_communities=pyg_data_train_f.community_id.max().item() + 1,
            hidden_dim=64, embed_dim=32, ablate=_ablate).to(device)
        optimizerB = torch.optim.Adam(modelB.parameters(), lr=0.001, weight_decay=5e-4)
        trainerB = HCGTrainer(modelB, optimizerB, ablate=_ablate)
        best_val_b, patience_b, best_state_b = float('inf'), 0, None
        for epoch in range(MAX_EPOCHS):
            train_loss_b = trainerB.train_epoch(pyg_data_train_f)
            val_loss_b = trainerB.evaluate_epoch(pyg_data_val_f)
            if epoch % 50 == 0:
                print(f"[fine] Epoch {epoch}: Train Loss = {train_loss_b:.4f} | Val Loss = {val_loss_b:.4f}")
            if val_loss_b < best_val_b:
                best_val_b, patience_b = val_loss_b, 0
                best_state_b = copy.deepcopy(modelB.state_dict())
                torch.save(best_state_b, MODEL_PATH_FINE)
            else:
                patience_b += 1
                if patience_b >= PATIENCE:
                    print(f"[fine] Early stopping triggered after {epoch} epochs. Best VAL loss: {best_val_b:.4f}")
                    break
        if best_state_b:
            modelB.load_state_dict(best_state_b)
        modelB.eval()
        print("[fine] fine-view training finished")

    print("\n--- Step 5: Evaluating with Distance ---")
    model.eval()

    print("\n--- Step 5.1: Get training embedding ---")
    with torch.no_grad():
        z_train, _ = model(
            pyg_data_train.x,
            pyg_data_train.edge_index,
            pyg_data_train.community_id,
            getattr(pyg_data_train, 'community_id_fine', None)
        )
    z_train_cpu = z_train.cpu().numpy()
    train_flow_mask = (pyg_data_train.y.cpu().numpy() != -1) 
    train_embeddings_flow = z_train_cpu[train_flow_mask]

    print("\n--- Step 5.1: Get community centroid ---")
    train_comm_ids = pyg_data_train.community_id.cpu().numpy()[train_flow_mask]
    community_centroids = {}
    unique_comm_ids = np.unique(train_comm_ids)

    for comm_id in unique_comm_ids:
        comm_mask = (train_comm_ids == comm_id)
        embeddings_in_comm = train_embeddings_flow[comm_mask]
        if embeddings_in_comm.shape[0] > 0:
            centroid = np.mean(embeddings_in_comm, axis=0)
            community_centroids[comm_id] = centroid      
    print(f"Distance Detector fitted on {len(community_centroids)} community centroids.")
    centroid_vectors = np.array(list(community_centroids.values()))

    start_time = time.time()
    print("\n--- Step 5.1: Get test embedding ---")
    with torch.no_grad():
        z_test, _ = model(
            pyg_data_test.x,
            pyg_data_test.edge_index,
            pyg_data_test.community_id,
            getattr(pyg_data_test, 'community_id_fine', None)
        )

    z_test_cpu = z_test.cpu().numpy()
    test_flow_mask = (pyg_data_test.y.cpu().numpy() != -1)
    test_embeddings_flow = z_test_cpu[test_flow_mask]
    test_true_labels = pyg_data_test.y.cpu().numpy()[test_flow_mask]
    end_time = time.time()
    print(f"test-set embedding time: {end_time-start_time}s")

    save_path = "./experiment_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_comm_ids = pyg_data_test.community_id.cpu().numpy()[test_flow_mask]

    np.savez_compressed(
        os.path.join(save_path, "manifold_data.npz"),

        train_embeddings=train_embeddings_flow,
        train_comm_ids=train_comm_ids,

        test_embeddings=test_embeddings_flow,
        test_labels=test_true_labels,
        test_comm_ids=test_comm_ids,

        centroids=centroid_vectors,
        centroid_comm_ids=unique_comm_ids
    )

    print(f"Manifold visualization data saved to {save_path}/manifold_data.npz")

    flow_meta_data = []
    for i in range(len(test_flows)):
        meta = test_flows[i]['meta']
        flow_meta_data.append({
            'src_ip': meta.get('src', 'N/A'),
            'dst_ip': meta.get('dst', 'N/A'),
            'timestamp': meta.get('timestamp', 'N/A'),
            'protocol': meta.get('protocol', 'N/A'),
            'file_id': meta.get('file_id', 'N/A'), 
            'label_name': 'Attack' if meta.get('label', 0) == 1 else 'Benign'
        })

    meta_df = pd.DataFrame(flow_meta_data)

    print("\n--- Step 5.3: Predicting ---")
    from scipy.spatial.distance import cdist
    distances = cdist(test_embeddings_flow, centroid_vectors, metric='euclidean')
    attack_scores = np.min(distances, axis=1) 
    np.savetxt(result_root + '/' + attack_pkl_name +"_min_test_distances.txt", attack_scores, fmt="%.4f")
    train_distances = cdist(train_embeddings_flow, centroid_vectors, metric='euclidean')

    comm_list = list(community_centroids.keys())
    col_of = {c: j for j, c in enumerate(comm_list)}
    comm_sizes = {c: int(np.sum(train_comm_ids == c)) for c in unique_comm_ids}
    min_train_distances = np.empty(len(train_embeddings_flow), dtype=np.float64)
    nearest_comm_train = np.empty(len(train_embeddings_flow), dtype=np.int64)
    for i in range(len(train_embeddings_flow)):
        ci = train_comm_ids[i]
        n_k = comm_sizes[ci]
        row = train_distances[i].copy()
        if n_k > 1:
            mu_loo = (community_centroids[ci] * n_k - train_embeddings_flow[i]) / (n_k - 1)
            row[col_of[ci]] = np.linalg.norm(train_embeddings_flow[i] - mu_loo)
        else:
            row[col_of[ci]] = np.inf
        j_min = int(np.argmin(row))
        min_train_distances[i] = row[j_min]
        nearest_comm_train[i] = j_min
    np.savetxt(result_root + '/' + attack_pkl_name +"_min_train_distances.txt", min_train_distances, fmt="%.4f")
    print(f"training distances (LOO): median={np.median(min_train_distances):.4f}, "
          f"Q95={np.quantile(min_train_distances, 0.95):.4f}")

    if args.ensemble and modelB is not None:
        print("\n--- Step 5.4: Dual-granularity score fusion ---")
        emb_trB = z_trB = None
        with torch.no_grad():
            z_trB, _ = modelB(pyg_data_train_f.x, pyg_data_train_f.edge_index,
                              pyg_data_train_f.community_id)
            z_teB, _ = modelB(pyg_data_test_f.x, pyg_data_test_f.edge_index,
                              pyg_data_test_f.community_id)
        z_trB = z_trB.cpu().numpy()
        z_teB = z_teB.cpu().numpy()
        emb_trB = z_trB[train_flow_mask]
        emb_teB = z_teB[test_flow_mask]

        train_comm_ids_f = pyg_data_train_f.community_id.cpu().numpy()[train_flow_mask]
        centroids_B = {c: emb_trB[train_comm_ids_f == c].mean(axis=0)
                       for c in np.unique(train_comm_ids_f)}
        cvec_B = np.array(list(centroids_B.values()))
        d_teB = cdist(emb_teB, cvec_B, metric='euclidean')
        sB_test = d_teB.min(axis=1)
        d_trB = cdist(emb_trB, cvec_B, metric='euclidean')
        sB_train = loo_min_distances(d_trB, emb_trB, train_comm_ids_f, centroids_B)
        print(f"fine view: {len(centroids_B)} centroids, training-distance median={np.median(sB_train):.4f}")

        scaleA = np.median(min_train_distances)
        scaleB = np.median(sB_train)
        sA_raw = attack_scores.copy()
        nA_te = sA_raw / scaleA
        nB_te = sB_test / scaleB
        fused_mean = (nA_te + nB_te) / 2.0
        fused_max = np.maximum(nA_te, nB_te)
        from sklearn.metrics import roc_auc_score as _auc
        print(f"[fusion] view-A AUC={_auc(test_true_labels, sA_raw):.4f} | "
              f"view-B AUC={_auc(test_true_labels, sB_test):.4f} | "
              f"fusion-mean AUC={_auc(test_true_labels, fused_mean):.4f} | "
              f"fusion-max AUC={_auc(test_true_labels, fused_max):.4f}")
        attack_scores = fused_mean
        attack_score_fused_max = fused_max
        attack_score_viewA = sA_raw
        attack_score_viewB = sB_test
        min_train_distances = (min_train_distances / scaleA + sB_train / scaleB) / 2.0
        print(f"fusion done: scaleA={scaleA:.4f}, scaleB={scaleB:.4f} | "
              f"fused training-distance median={np.median(min_train_distances):.4f}")

    THRESHOLD_Q = 0.95 

    _DATASET_EVT_PARAMS = {
        'botnet2014': (0.975, 0.002),
        'ids2017':    (0.975,  0.002),
        'unsw':       (0.975, 0.002),
        'ustc':       (0.975, 0.002),
    }
    initial_q, q_risk = _DATASET_EVT_PARAMS.get(dataset, (0.975, 0.002))

    with torch.no_grad():
        z_val, _ = model(pyg_data_val.x, pyg_data_val.edge_index,
                         pyg_data_val.community_id,
                         getattr(pyg_data_val, 'community_id_fine', None))
    val_flow_mask = (pyg_data_val.y.cpu().numpy() != -1)
    val_embeddings_flow = z_val.cpu().numpy()[val_flow_mask]
    val_distances = cdist(val_embeddings_flow, centroid_vectors, metric='euclidean')
    min_val_distances = val_distances.min(axis=1)
    np.savetxt(result_root + '/' + attack_pkl_name + "_min_val_distances.txt",
               min_val_distances, fmt="%.4f")
    print(f"validation distances: median={np.median(min_val_distances):.4f}, "
          f"Q95={np.quantile(min_val_distances, 0.95):.4f}")
    if args.calib_on == 'val' and not args.ensemble:
        evt_src, src_name = min_val_distances, 'val'
    elif args.calib_on == 'val' and args.ensemble:
        print("note: in ensemble mode the val calibration source mismatches the fused scores, falling back to train-LOO")
        evt_src, src_name = min_train_distances, 'train-LOO'
    else:
        evt_src, src_name = min_train_distances, 'train-LOO'

    T_evt = get_evt_threshold(evt_src, initial_quantile=initial_q, risk_prob=q_risk)
    T_train_calib = get_evt_threshold(min_train_distances, initial_quantile=initial_q, risk_prob=q_risk)
    T_val_calib = get_evt_threshold(min_val_distances, initial_quantile=initial_q, risk_prob=q_risk)

    print(f"Base Threshold (Q={initial_q}): {np.quantile(min_train_distances, initial_q):.4f}")
    print(f"EVT Dynamic Threshold (Risk={q_risk}, source={src_name}): T={T_evt:.4f}")

    for _src_name, _T in [('train-LOO', T_train_calib), ('val', T_val_calib)]:
        _p = (attack_scores > _T).astype(int)
        from sklearn.metrics import f1_score as _f1
        print(f"[calib-compare] source={_src_name:9s} T={_T:.4f} "
              f"attack-F1={_f1(test_true_labels, _p, pos_label=1, zero_division=0):.4f} "
              f"FPR={( _p[test_true_labels == 0] == 1).mean():.4f}")

    from sklearn.metrics import f1_score as _f1
    def _per_comm_thresholds(calib_scores, calib_assign, n_min=100):
        T_glob = get_evt_threshold(calib_scores, initial_quantile=initial_q, risk_prob=q_risk)
        T_map = {}
        for c in comm_list:
            s_k = calib_scores[calib_assign == c]
            if len(s_k) >= n_min:
                try:
                    T_map[c] = get_evt_threshold(s_k, initial_quantile=initial_q, risk_prob=q_risk)
                except Exception:
                    T_map[c] = T_glob
            else:
                T_map[c] = T_glob
        return T_map, T_glob

    comm_arr = np.array(comm_list)
    test_nearest = comm_arr[distances.argmin(axis=1)]
    val_nearest = comm_arr[val_distances.argmin(axis=1)]
    tr_assign = comm_arr[nearest_comm_train]

    comm_thresholds = {}
    for _src_name2, _src_scores, _src_assign in [
            ('comm(train-LOO)', min_train_distances, tr_assign),
            ('comm(val)', min_val_distances, val_nearest)]:
        T_map, T_glob = _per_comm_thresholds(_src_scores, _src_assign)
        T_per_flow = np.array([T_map[c] for c in test_nearest])
        _p = (attack_scores > T_per_flow).astype(int)
        n_fallback = sum(1 for c in comm_list if T_map[c] == T_glob)
        print(f"[calib-compare] source={_src_name2:14s} "
              f"attack-F1={_f1(test_true_labels, _p, pos_label=1, zero_division=0):.4f} "
              f"FPR={(_p[test_true_labels == 0] == 1).mean():.4f} "
              f"(fallback communities {n_fallback}/{len(comm_list)})")
        comm_thresholds[_src_name2] = T_map

    start_time = time.time()
    if args.threshold_mode == 'quantile':
        T_used = np.quantile(min_train_distances, 0.995)
        print(f"Fixed Quantile Threshold (Q=0.995): T={T_used:.4f}")
    elif args.threshold_mode.startswith('comm_evt'):
        if args.ensemble:
            print("note: in ensemble mode scores are fused and per-community thresholds do not apply, falling back to global EVT")
            T_used = T_evt
        else:
            _key = 'comm(val)' if args.threshold_mode == 'comm_evt_val' else 'comm(train-LOO)'
            T_used = np.array([comm_thresholds[_key][c] for c in test_nearest])
            print(f"Per-Community EVT Threshold (source={_key}, "
                  f"per-flow T median={np.median(T_used):.4f})")
    else:
        T_used = T_evt
    binary_preds = (attack_scores > T_used).astype(int)
    end_time = time.time()
    print(f"final EVT test time {end_time-start_time}s over {len(binary_preds)} samples")

    original_decision_score = -attack_scores

    data = {
        "sample_index": np.arange(len(attack_scores)),
        "true_label": test_true_labels,
        "pred_label": binary_preds,
        "attack_score": attack_scores,
        "original_decision_score": original_decision_score,
        "community_id": pyg_data_test.community_id.cpu().numpy()[test_flow_mask]
    }
    if args.ensemble and modelB is not None:
        data["attack_score_viewA"] = attack_score_viewA
        data["attack_score_viewB"] = attack_score_viewB
        data["attack_score_fused_max"] = attack_score_fused_max
    df = pd.DataFrame(data)

    final_df = pd.concat([df, meta_df], axis=1)
    csv_path = result_root + '/' + attack_pkl_name + "_distance.csv"
    final_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        float_format="%.4f"
    )

    print("\n--- Classification Report on Test Set (Distance Detector) ---")
    report = classification_report(
        test_true_labels, 
        binary_preds, 
        labels=[0, 1],
        target_names=['Benign', 'Attack'],
        zero_division=0,
        digits=4
    )
    print(report)

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_val = roc_auc_score(test_true_labels, attack_scores)
    auprc_val = average_precision_score(test_true_labels, attack_scores)
    metric_line = (f"AUC={auc_val:.4f}  AUPRC={auprc_val:.4f}  "
                   f"(attack base rate={test_true_labels.mean():.4f}, "
                   f"threshold_mode={args.threshold_mode}, ensemble={args.ensemble})")
    print(metric_line)

    txt_file = result_root + '/' + attack_pkl_name + "_distance.txt"
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report + "\n" + metric_line + "\n")
    except Exception as e:
        print(f"\nfailed to write classification report to file: {e}")

    if args.sgv:
        print("\n--- Step 6: SGV (Semantic-Group Verification) ---")
        from collections import defaultdict as _dd
        _fine = {}
        for _tag, _builder, _pyg in [('train', builder_train, pyg_data_train),
                                     ('val', builder_val, pyg_data_val),
                                     ('test', builder_test, pyg_data_test)]:
            _fmap = OptimizedGraphBuilder._refine_communities(
                _builder.G, _builder.partition_coarse, _pyg.x.cpu().numpy())
            _fine[_tag] = np.array([_fmap[i] for i in range(len(_fmap))])
            print(f"[SGV] {_tag} fine subgroups={len(np.unique(_fine[_tag]))}")

        def _sgv_split(_pyg, _fine_ids):
            _y = _pyg.y.cpu().numpy()
            _X = _pyg.x.cpu().numpy()
            _m = _y != -1
            _pos = np.where(_m)[0]
            _g = _dd(list)
            for _i_new, _i_orig in enumerate(_pos):
                _g[int(_fine_ids[_i_orig])].append(_i_new)
            return _X[_m], _g

        Xtr, gtr = _sgv_split(pyg_data_train, _fine['train'])
        Xva, gva = _sgv_split(pyg_data_val, _fine['val'])
        Xte, gte = _sgv_split(pyg_data_test, _fine['test'])

        def _proto(X, g):
            return {k: X[np.array(v)].mean(axis=0)
                    for k, v in g.items() if len(v) >= args.sgv_min_k}

        Rtr = np.array(list(_proto(Xtr, gtr).values()))
        Pva, Pte = _proto(Xva, gva), _proto(Xte, gte)
        if len(Pva) > 0 and len(Rtr) > 0:
            d_va = np.array([np.linalg.norm(Rtr - v, axis=1).min() for v in Pva.values()])
            tail = float(np.quantile(d_va, 1 - args.sgv_q))
        else:
            d_va, tail = np.array([0.0]), np.inf
        thr_B = args.sgv_c * tail
        print(f"[SGV] val subgroup null distribution: n={len(d_va)}, (1-q)-quantile={tail:.4f}, "
              f"verdict threshold T_B=C*quantile={thr_B:.4f} (C={args.sgv_c}, q={args.sgv_q}, min_k={args.sgv_min_k})")

        d_te = {k: float(np.linalg.norm(Rtr - v, axis=1).min()) for k, v in Pte.items()}
        d_node = np.zeros(len(test_true_labels))
        for k, d in d_te.items():
            d_node[gte[k]] = d
        flag_B = d_node > thr_B

        T_base = T_used if np.isscalar(T_used) else T_evt
        _base_pred = (attack_scores > T_base).astype(int)
        fused_pred = (_base_pred == 1) | flag_B
        chA = attack_scores / T_base
        chB = d_node / thr_B if np.isfinite(thr_B) and thr_B > 0 else np.zeros_like(d_node)
        fused_score = np.maximum(chA, chB)

        from sklearn.metrics import f1_score as _f1s, roc_auc_score as _aucs, average_precision_score as _aps
        _p = fused_pred.astype(int)
        _n_flagged = int(sum(1 for d in d_te.values() if d > thr_B))
        print(f"[SGV] Channel B: candidate subgroups={len(d_te)}, flagged={_n_flagged}, "
              f"new alerts={int((_p & (test_true_labels == 0)).sum() - (_base_pred == 1)[test_true_labels == 0].sum())} FP, "
              f"new hits={int(((_p == 1) & (_base_pred == 0) & (test_true_labels == 1)).sum())} TP")
        _fpr = (_p[test_true_labels == 0] == 1).mean()
        print(f"[SGV] fused result: F1={_f1s(test_true_labels, _p, pos_label=1, zero_division=0):.4f} "
              f"FPR={_fpr:.4f} | AUC={_aucs(test_true_labels, fused_score):.4f} "
              f"AUPRC={_aps(test_true_labels, fused_score):.4f}")
        print(f"[SGV] vs std: F1={_f1s(test_true_labels, _base_pred, pos_label=1, zero_division=0):.4f} "
              f"AUC={_aucs(test_true_labels, chA):.4f}")

        _sgv_df = pd.DataFrame({
            "sample_index": np.arange(len(attack_scores)),
            "true_label": test_true_labels,
            "attack_score": attack_scores,
            "sgv_d": d_node, "sgv_flag": flag_B.astype(int),
            "std_pred": _base_pred, "fused_pred": _p, "fused_score": fused_score,
        })
        _sgv_df.to_csv(result_root + '/' + attack_pkl_name + "_sgv_distance.csv",
                       index=False, float_format="%.4f")
        print(f"[SGV] results saved: {result_root}/{attack_pkl_name}_sgv_distance.csv")

if __name__ == "__main__":
    main()
