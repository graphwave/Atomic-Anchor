import numpy as np
import scipy.stats
from scapy.all import rdpcap, IP, TCP, UDP
from scapy.packet import Raw
from scapy.all import Packet
import os
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class AdvancedFeatureExtractor:
    def __init__(self):
        self.feature_dim = 18 

    def calculate_entropy(self, data):
        if len(data) == 0: return 0
        counts = np.bincount(data)
        probs = counts[counts > 0] / len(data)
        return -np.sum(probs * np.log2(probs))

    def extract_flow_features(self, packets):
        if len(packets) < 3: return None

        has_nonzero_payload = False
        for p in packets:

            if IP in p and (TCP in p[IP] or UDP in p[IP]):
                transport_layer = p[IP][TCP] if TCP in p[IP] else p[IP][UDP]

                if Raw in transport_layer and len(transport_layer.payload) > 0:
                    has_nonzero_payload = True
                    break

        if not has_nonzero_payload:
            return None

        lengths = [len(p) for p in packets]

        timestamps = [float(p.time) for p in packets]
        iats = np.diff(timestamps)

        features = []

        features.append(np.min(lengths))
        features.append(np.max(lengths))
        features.append(np.mean(lengths))
        features.append(np.std(lengths))
        features.append(np.percentile(lengths, 25))
        features.append(np.percentile(lengths, 75))
        features.append(self.calculate_entropy(lengths))
        features.append(scipy.stats.skew(lengths))

        if len(iats) > 0:
            features.append(np.min(iats))
            features.append(np.max(iats))
            features.append(np.mean(iats))
            features.append(np.std(iats))
            features.append(self.calculate_entropy((iats * 1000).astype(int)))
        else:
            features.extend([0]*5)

        features.append(len(packets))
        features.append(sum(lengths))
        features.append(timestamps[-1] - timestamps[0])

        has_tcp = 1 if TCP in packets[0] else 0
        features.append(has_tcp)

        return np.array(features, dtype=np.float32)

    def process_pcap(self, file_path, label):
        try:
            packets = rdpcap(file_path)

            ip_packets = [p for p in packets if IP in p]
            if not ip_packets: return None

            feats = self.extract_flow_features(ip_packets)
            if feats is None: return None

            first = ip_packets[0]
            meta = {
                'src': first[IP].src,
                'dst': first[IP].dst,
                'start_time': float(ip_packets[0].time),
                'end_time': float(ip_packets[-1].time),
                'file_id': os.path.basename(file_path),
                'raw_lengths': [len(p) for p in ip_packets],
                'label': label
            }
            return {'x': feats, 'meta': meta}
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def _process_single_file(args):
    file_path, label = args
    local_extractor = AdvancedFeatureExtractor() 
    return local_extractor.process_pcap(file_path, label)

def load_or_extract_features(extractor, data_root='data/', save_path='features_cache.pkl'):

    if os.path.exists(save_path):
        print(f"[{save_path}] found. Loading features from file...")
        try:
            with open(save_path, 'rb') as f:
                processed_data = pickle.load(f)
            print(f"Successfully loaded {len(processed_data)} flows from cache.")
            return processed_data
        except Exception as e:
            print(f"Error loading cache: {e}. Re-extracting...")

    print("Cache not found. Extracting features from PCAP files...")

    benign_path = os.path.join(data_root, "benign")
    attack_path = os.path.join(data_root, "attack")

    if not os.path.exists(benign_path) or not os.path.exists(attack_path):
        print(f"Error: Data directories not found in {data_root}")
        return []

    tasks = []

    benign_files = [f for f in os.listdir(benign_path) if f.endswith('.pcap')]
    for f in benign_files:
        tasks.append((os.path.join(benign_path, f), 0))

    attack_files = [f for f in os.listdir(attack_path) if f.endswith('.pcap')]
    for f in attack_files:
        tasks.append((os.path.join(attack_path, f), 1))

    if not tasks:
        print("No PCAP files found for processing.")
        return []

    num_processes = min(64, cpu_count())
    print(f"Starting parallel feature extraction using {num_processes} processes for {len(tasks)} files...")

    processed_data = []

    try:
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_file, tasks),
                total=len(tasks),
                desc="Total Feature Extraction Progress"
            ))

        processed_data = [item for item in results if item is not None]

    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")

    print(f"Extraction finished. Saving {len(processed_data)} flows to {save_path}...")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print("Cache saved successfully.")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return processed_data

def extract_single_category_to_pkl(
    category_dir,
    label,
    save_pkl_dir=".",
    num_processes=64
):

    if not os.path.isdir(category_dir):
        print(f"Error: Category directory {category_dir} does not exist!")
        return None

    category_name = os.path.basename(category_dir.rstrip('/'))
    dir_name = os.path.dirname(category_dir)
    pkl_file_path = os.path.join(save_pkl_dir, f"{category_name}.pkl")

    pcap_files = [f for f in os.listdir(category_dir) if f.endswith('.pcap')]
    if not pcap_files:
        print(f"Warning: No PCAP files found in {category_dir}")
        return None
    print(f"Found {len(pcap_files)} PCAP files in {category_dir}")

    tasks = []
    for f in pcap_files:
        file_path = os.path.join(category_dir, f)
        tasks.append((file_path, label))

    num_processes = min(num_processes, cpu_count())
    print(f"Starting parallel extraction for {category_name} using {num_processes} processes...")

    processed_results = []
    try:
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_file, tasks),
                total=len(tasks),
                desc=f"Extracting {category_name} features"
            ))

        processed_results = [item for item in results if item is not None]
    except Exception as e:
        print(f"Error during parallel extraction: {e}")
        return None

    if not processed_results:
        print(f"Warning: No valid features extracted from {category_dir}")
        return None

    print(f"Extraction finished. Saving {len(processed_results)} flows to {pkl_file_path}...")
    try:
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(processed_results, f)
        print(f"Successfully saved {category_name} features to {pkl_file_path}")
    except Exception as e:
        print(f"Failed to save {category_name} PKL file: {e}")
        return None

    return pkl_file_path

if __name__ == "__main__":
    dataset = 'unsw'

    single_category_path = extract_single_category_to_pkl(

        category_dir=f"/data/fp/TrafficScope-modified/middleResults/pcap_data/unsw/reconnaissance_sessions",
        label=1,
        save_pkl_dir="data/unsw/",
        num_processes=64
    )

    if single_category_path and os.path.exists(single_category_path):
        with open(single_category_path, 'rb') as f:
            single_category_data = pickle.load(f)
        print(f"\nValidation: {single_category_path} contains {len(single_category_data)} valid records")
        print(f"Example record keys: {list(single_category_data[0].keys())}")
