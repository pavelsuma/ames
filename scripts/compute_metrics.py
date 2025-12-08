import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def parse_label(path):
    """
    DEFINES WHAT A CLASS IS.
    Change this logic based on your filename format.
    
    Example 1: Path is "dog/image_01.jpg" -> Class is "dog"
    Example 2: Path is "n01443537_wolf.jpg" -> Class is "n01443537"
    """
    # Logic: Take the parent folder name as the class
    # e.g. "bold_bimp_000/query/Q000_00.jpg" -> Label might be "bold_bimp_000"
    
    parts = path.split('/')
    
    # ADJUST THIS INDEX based on your folder structure!
    # If path is "data/ilias/class_name/img.jpg", usually the class is the folder.
    # Here we try to find a meaningful folder name.
    if len(parts) > 1:
        return parts[0] # Returns 'bold_bimp_000' based on your previous logs
    return path # Fallback

def compute_ap(ranked_indices, ground_truth_indices):
    """
    Computes Average Precision (AP) for a single query.
    """
    if len(ground_truth_indices) == 0:
        return 0.0

    intersect_size = 0
    old_recall = 0.0
    ap = 0.0
    
    ground_truth_set = set(ground_truth_indices)
    
    for i, idx in enumerate(ranked_indices):
        if idx in ground_truth_set:
            intersect_size += 1
            
            precision = intersect_size / (i + 1)
            recall = intersect_size / len(ground_truth_indices)
            
            ap += precision * (recall - old_recall)
            old_recall = recall
            
    return ap

def load_list(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            # Clean up line "path/img.jpg,0,500..." -> "path/img.jpg"
            clean_path = line.strip().split(',')[0].split(' ')[0]
            data.append(clean_path)
    return data

def main():
    parser = argparse.ArgumentParser()
    # Paths from your previous commands
    parser.add_argument('--dataset', default='ilias')
    parser.add_argument('--results_pkl', default='data/ilias/ranked_dinov3_reranked_dinov2.pkl')
    parser.add_argument('--query_list', default='data/ilias/test_query.txt')
    parser.add_argument('--gallery_list', default='data/ilias/test_gallery.txt')
    args = parser.parse_args()

    # 1. Load Filenames
    print(">>> Loading filenames...")
    q_paths = load_list(args.query_list)
    g_paths = load_list(args.gallery_list)

    # 2. Build Ground Truth (The "Oracle")
    # Map: Query_Index -> List of Matching Gallery Indices
    print(">>> Building Ground Truth based on Folder Names...")
    
    # Pre-calculate gallery labels
    g_labels = [parse_label(p) for p in g_paths]
    
    # Map label string to list of gallery indices
    # e.g. {'dog': [0, 5, 10], 'cat': [1, 2]}
    label_to_g_indices = {}
    for idx, label in enumerate(g_labels):
        if label not in label_to_g_indices:
            label_to_g_indices[label] = []
        label_to_g_indices[label].append(idx)

    # 3. Load Predictions
    print(f">>> Loading Results: {args.results_pkl}")
    with open(args.results_pkl, 'rb') as f:
        data = pickle.load(f)
        
    # Handle dict/list variations
    if isinstance(data, dict):
        pred_indices = data['indices']
    else:
        # Assuming [scores, indices] list
        pred_indices = data[1]

    # Ensure integer
    if not isinstance(pred_indices, np.ndarray):
        pred_indices = np.array(pred_indices)
    pred_indices = pred_indices.astype(int)

    # 4. Compute metrics
    aps = []
    
    print(f">>> Calculating AP for {len(q_paths)} queries...")
    print(f"{'idx':<5} | {'Label':<20} | {'AP':<10}")
    print("-" * 40)

    for i, q_path in enumerate(q_paths):
        q_label = parse_label(q_path)
        
        # Get correct answers for this query
        positive_g_indices = label_to_g_indices.get(q_label, [])
        
        # Get predicted answers (row i)
        ranked_list = pred_indices[i]
        
        # Calculate AP
        ap = compute_ap(ranked_list, positive_g_indices)
        aps.append(ap)
        
        # Print first 10 for sanity check
        if i < 10:
            print(f"{i:<5} | {q_label:<20} | {ap:.4f}")

    mAP = np.mean(aps)
    print("-" * 40)
    print(f"Total mAP@{pred_indices.shape[1]}: {mAP:.4f} ({mAP*100:.2f}%)")

if __name__ == "__main__":
    main()