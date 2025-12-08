import os
import argparse
import pickle
import h5py
import torch
import json
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Run AMES Inference (Fixed Slicing)')
    parser.add_argument('--dataset', type=str, default='ilias')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--global_desc', type=str, default='siglip2')
    parser.add_argument('--local_desc', type=str, default='dinov2')
    parser.add_argument('--topk', type=int, default=100)
    
    args = parser.parse_args()

    # --- Setup Paths ---
    dataset_dir = os.path.join(args.data_root, args.dataset)
    rank_path = os.path.join(dataset_dir, f'nn_{args.global_desc}.pkl')
    query_h5_path = os.path.join(dataset_dir, f'{args.local_desc}_query_local.hdf5')
    gallery_h5_path = os.path.join(dataset_dir, f'{args.local_desc}_gallery_local.hdf5')
    
    output_pkl_path = os.path.join(dataset_dir, f'ranked_{args.global_desc}_reranked_{args.local_desc}.pkl')
    output_json_path = os.path.join(dataset_dir, f'ranked_{args.global_desc}_reranked_{args.local_desc}.json')

    # --- Load Global Rankings ---
    print(f"Loading global ranking from: {rank_path}")
    try:
        with open(rank_path, 'rb') as f:
            initial_ranks_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # Handle Tensor vs Numpy vs List format
    target_data = initial_ranks_data
    if isinstance(initial_ranks_data, dict):
        # If dict, try to find standard keys
        if 'scores' in initial_ranks_data and 'indices' in initial_ranks_data:
            target_data = [initial_ranks_data['scores'], initial_ranks_data['indices']]
    
    # Normalize to List of Arrays [Scores, Indices]
    if isinstance(target_data, torch.Tensor):
        target_data = target_data.cpu().numpy()
    
    # If it's a tuple/list, ensure contents are numpy
    final_data_list = []
    if isinstance(target_data, (list, tuple)):
        for item in target_data:
            if torch.is_tensor(item):
                final_data_list.append(item.cpu().numpy())
            else:
                final_data_list.append(np.array(item))
    else:
        # Fallback for weird structures (assumes matrix)
        target_data = np.array(target_data)
        final_data_list = [target_data[0], target_data[1]]

    global_scores = final_data_list[0]
    global_indices = final_data_list[1]

    # --- CRITICAL FIX: Ensure Indices are Integers ---
    # (Handles the 724.0 float issue)
    global_indices = global_indices.astype(np.int64)

    num_queries = global_indices.shape[0]
    print(f"Loaded {num_queries} queries to process.")
    
    # --- Load AMES Model ---
    print("Loading AMES model (Binary Distilled)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = torch.hub.load('pavelsuma/ames', 'dinov2_ames_dist').to(device).eval()
    except:
        # Fallback if hub fails
        print("Hub load failed. Trying local 'dinov2_ames.pt'...")
        state = torch.load('dinov2_ames.pt')
        if isinstance(state, dict): state = state['model']
        model = state.to(device).eval()

    print(f"Opening local feature files...")
    reranked_scores = []
    reranked_indices = []

    # Open files safely
    with h5py.File(query_h5_path, 'r') as q_h5, h5py.File(gallery_h5_path, 'r') as g_h5:
        
        q_feats_dset = q_h5['features']
        g_feats_dset = g_h5['features']

        print(f"Reranking top-{args.topk} candidates...")
        
        for q_idx in tqdm(range(num_queries)):
            
            # --- 1. Get Query Local Features ---
            try:
                # Direct Indexing: gets (N, Dim) -> e.g. (700, 773)
                q_data = torch.from_numpy(q_feats_dset[q_idx]) 

                # --- FIX: SLICE GEOMETRY ---
                # Skip the first 5 elements (Geometry), keep the rest (Descriptors)
                q_local = q_data[:, 5:].to(device)

                q_local = q_local.unsqueeze(0) # Shape (1, N, D)
                
            except Exception as e:
                print(f"\n[Error] Query {q_idx} failed: {e}")
                # Fallback: keep original order
                reranked_scores.append(np.zeros(args.topk)) 
                reranked_indices.append(global_indices[q_idx, :args.topk])
                continue

            # --- 2. Identify Candidates ---
            candidate_indices = global_indices[q_idx, :args.topk]
            
            # --- 3. Process Candidates ---
            ames_scores_q = []
            
            # Optimization: You could batch this, but loop is safer for varied shapes
            for c_idx in candidate_indices:
                try:
                    c_data = torch.from_numpy(g_feats_dset[c_idx])
                    
                    # --- FIX: SLICE GEOMETRY ---
                    c_feat = c_data[:, 5:].to(device)
                    
                    c_input = c_feat.unsqueeze(0)
                    
                    with torch.no_grad():
                        sim = model(src_local=q_local, tgt_local=c_input)
                        ames_scores_q.append(sim.item())
                        
                except Exception as e:
                    # If fetching fails, give low score
                    ames_scores_q.append(-999.0)

            ames_scores_q = np.array(ames_scores_q)

            # --- 4. Re-Sort based ONLY on AMES score ---
            # argsort sorts ascending, so [::-1] makes it descending (highest score first)
            sorted_order = np.argsort(ames_scores_q)[::-1]
            
            final_indices = candidate_indices[sorted_order]
            final_scores = ames_scores_q[sorted_order]
            
            reranked_scores.append(final_scores)
            reranked_indices.append(final_indices)

    # --- Save Outputs ---
    output_data_pkl = {
        'scores': np.array(reranked_scores),
        'indices': np.array(reranked_indices)
    }
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data_pkl, f)
    print(f"\nSaved Pickle: {output_pkl_path}")

    output_data_json = {
        'scores': np.array(reranked_scores).tolist(),
        'indices': np.array(reranked_indices).tolist()
    }
    with open(output_json_path, 'w') as f:
        json.dump(output_data_json, f, indent=4)
    print(f"Saved JSON:   {output_json_path}")

if __name__ == "__main__":
    main()