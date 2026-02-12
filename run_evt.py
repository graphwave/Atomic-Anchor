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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

def read_pkl_file(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"Warning: PKL file {pkl_path} does not exist, skipping reading")
        return []
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if 'ids2017' in pkl_path:
            valid_data = [item for item in data if item is not None and item['x'][13] > 3 and item['x'][15] < 50 and item['x'][16] >0]
            print(f"Successfully read {pkl_path}, total valid data: {len(valid_data)} entries")
            return valid_data
        else:
            return data
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}, skipping reading")
        return []

def main():
    extractor = AdvancedFeatureExtractor()
    dataset = 'botnet2014'
    data_root = f'data/{dataset}'
    result_root = 'result_pe/result_' + dataset
    processed_data = []
    os.makedirs(result_root, exist_ok=True)
    
    benign_pkl_path = os.path.join(data_root, "benign.pkl")
    attack_pkl_name ="Weasel.pkl"
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
    
    builder_train = OptimizedGraphBuilder(time_win=1, corr_th=0.8)
    pyg_data_train = builder_train.build(train_flows) 
    
    builder_val = OptimizedGraphBuilder(time_win=1, corr_th=0.8) 
    pyg_data_val = builder_val.build(val_flows) 
    
    builder_test = OptimizedGraphBuilder(time_win=1, corr_th=0.8) 
    pyg_data_test = builder_test.build(test_flows)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    pyg_data_train = pyg_data_train.to(device)
    pyg_data_val = pyg_data_val.to(device)
    pyg_data_test = pyg_data_test.to(device)
    
    num_communities_train = pyg_data_train.community_id.max().item() + 1
    FEATURE_DIM = 17 
    
    model = CommunityGuidedGAT(
        feature_dim=FEATURE_DIM, 
        num_communities=num_communities_train,
        hidden_dim=64, 
        embed_dim=32
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    trainer = HCGTrainer(model, optimizer)

    print("\n--- Step 4: Starting Contrastive Training with Validation ---")
    PATIENCE = 800 
    MAX_EPOCHS = 1000
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None 
    MODEL_PATH = 'model/model_' + dataset + '.pth'
    
    if os.path.exists(MODEL_PATH):
        print(f"Detected existing local model file: {MODEL_PATH}, skipping training and loading directly...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        need_training = False 
    else:
        print("No pre-trained model found locally, preparing to start training...")
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

    print("\n--- Step 5: Evaluating with Distance ---")
    model.eval()
    
    print("\n--- Step 5.1: Get training embedding ---")
    with torch.no_grad():
        z_train, _ = model(
            pyg_data_train.x, 
            pyg_data_train.edge_index, 
            pyg_data_train.community_id
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
    
    print("\n--- Step 5.1: Get test embedding ---")
    with torch.no_grad():
        z_test, _ = model(
            pyg_data_test.x, 
            pyg_data_test.edge_index, 
            pyg_data_test.community_id
        )
    
    z_test_cpu = z_test.cpu().numpy()
    test_flow_mask = (pyg_data_test.y.cpu().numpy() != -1)
    test_embeddings_flow = z_test_cpu[test_flow_mask]
    test_true_labels = pyg_data_test.y.cpu().numpy()[test_flow_mask]

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
    distances = cdist(test_embeddings_flow, centroid_vectors, metric='euclidean')
    attack_scores = np.min(distances, axis=1) 
    np.savetxt(result_root + '/' + attack_pkl_name +"_min_test_distances.txt", attack_scores, fmt="%.4f")
    train_distances = cdist(train_embeddings_flow, centroid_vectors, metric='euclidean')
    min_train_distances = np.min(train_distances, axis=1)
    np.savetxt(result_root + '/' + attack_pkl_name +"_min_train_distances.txt", min_train_distances, fmt="%.4f")

    THRESHOLD_Q = 0.95 
    T_threshold = np.quantile(min_train_distances, THRESHOLD_Q)
    print(f"Anomaly Threshold (Q={THRESHOLD_Q}) determined from training distance: T={T_threshold:.4f}")

    T_threshold1 = np.quantile(min_train_distances, 0.99)
    print(f"Anomaly Threshold (Q=0.99) determined from training distance: T={T_threshold1:.4f}")

    T_threshold2 = np.quantile(min_train_distances, 0.9)
    print(f"Anomaly Threshold (Q=0.9) determined from training distance: T={T_threshold2:.4f}")

    initial_q = 0.95
    q_risk =  0.002

    T_evt = get_evt_threshold(min_train_distances, initial_quantile=initial_q, risk_prob=q_risk)

    print(f"Base Threshold (Q={initial_q}): {np.quantile(min_train_distances, initial_q):.4f}")
    print(f"EVT Dynamic Threshold (Risk={q_risk}): T={T_evt:.4f}")

    binary_preds = (attack_scores > T_evt).astype(int)
    original_decision_score = -attack_scores
    
    data = {
        "sample_index": np.arange(len(attack_scores)),
        "true_label": test_true_labels,
        "pred_label": binary_preds,
        "attack_score": attack_scores,
        "original_decision_score": original_decision_score,
        "community_id": pyg_data_test.community_id.cpu().numpy()[test_flow_mask]
    }
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
    
    txt_file = result_root + '/' + attack_pkl_name + "_distance.txt"
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report)
    except Exception as e:
        print(f"\nFailed to write classification report to file: {e}")

if __name__ == "__main__":
    main()
