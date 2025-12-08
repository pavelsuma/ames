import argparse
import os.path
from glob import glob

import h5py
import os.path as osp
import numpy as np
import torch

from src.utils.utils import pickle_save, pickle_load
from src.utils.revisited import compute_metrics


def load_or_combine(feat_dir, desc_name, file_type, global_type_hint):
    """
    Smart loader that tries to find the file with 'cls' or 'global' suffix.
    """
    file_type_str = '_' + file_type if file_type is not None else ''
    
    # Define candidate filenames (try the hinted type first, then the other)
    types_to_try = [global_type_hint]
    if global_type_hint == 'cls':
        types_to_try.append('global')
    else:
        types_to_try.append('cls')

    final_desc = None
    found_name = None

    for g_type in types_to_try:
        file_name = f'{desc_name}{file_type_str}_{g_type}.hdf5'
        full_path = os.path.join(feat_dir, file_name)

        if os.path.exists(full_path):
            print(f"Loading found file: {file_name}")
            with h5py.File(full_path, 'r') as f:
                final_desc = f['features'][:]
            found_name = file_name
            break
        
        # Check for split files (e.g. siglip2_xa0_global.hdf5)
        splits = sorted(glob(osp.join(feat_dir, f'{desc_name}_xa?_{g_type}.hdf5')))
        if len(splits):
            print(f"Found split files for type '{g_type}'. Combining...")
            images_per_hdf5 = []
            for h in splits:
                with h5py.File(os.path.join(feat_dir, h), 'r') as f:
                    images_per_hdf5.append(f['features'][:])
            final_desc = np.concatenate(images_per_hdf5)
            
            # Save combined file for next time
            with h5py.File(full_path, "w") as f:
                f.create_dataset("features", data=final_desc, dtype=final_desc.dtype)
            found_name = file_name
            break

    if final_desc is None:
        raise FileNotFoundError(f"Could not find features for {desc_name} in {feat_dir}. \n"
                                f"Checked for both '_cls.hdf5' and '_global.hdf5'.")

    return final_desc


def main():
    parser = argparse.ArgumentParser(description='Compute and store image similarities and ranking with global descriptors.')
    parser.add_argument('--dataset', help='Dataset name to load descriptors of.')
    parser.add_argument('--desc_name', default='siglip2', help='Descriptors to load based on name.')
    parser.add_argument('--data_root', default='data', help='Root folder of the saved descriptor sets.')
    parser.add_argument('--shortlist_size', default=1600, type=int, help='Number of nearest neighbors of each sample.')

    args = parser.parse_args()
    dataset = args.dataset
    feat_dir = os.path.join(args.data_root, dataset)
    desc_name = args.desc_name
    m = args.shortlist_size
    
    # Default hint: dinov2 usually uses 'cls', others usually 'global'.
    # Our smart loader will try both anyway.
    global_type_hint = 'cls' if 'dinov2' in desc_name else 'global'
    
    desc, gnd, query = None, None, None

    if dataset == 'gldv2':
        desc = load_or_combine(feat_dir, desc_name, None, global_type_hint)
        query = desc

    elif dataset == 'revisitop1m' or dataset[-3:] == '+1m':
        desc = load_or_combine(osp.join(args.data_root, 'revisitop1m'), desc_name, None, global_type_hint)
        query = desc

    # --- ILIAS SUPPORT ---
    elif dataset == 'ilias':
        print(f"Processing ILIAS with descriptor: {desc_name}")
        # Load Query
        query = load_or_combine(feat_dir, desc_name, 'query', global_type_hint)
        # Load Gallery
        desc = load_or_combine(feat_dir, desc_name, 'gallery', global_type_hint)
        
        m = len(desc) # Set k to full database size for full ranking

    elif dataset.startswith(('roxford5k', 'rparis6k', 'gldv2-test')):
        small_feat_dir = os.path.join(args.data_root, dataset.split('+')[0])
        gnd = pickle_load(osp.join(feat_dir, f'gnd_{dataset}.pkl'))['gnd']

        query = load_or_combine(small_feat_dir, desc_name, 'query', global_type_hint)
        db_desc = load_or_combine(small_feat_dir, desc_name, 'gallery', global_type_hint)

        if desc is not None:
            desc = np.concatenate((db_desc, desc))
            with h5py.File(os.path.join(feat_dir, f'{desc_name}_gallery_{global_type_hint}.hdf5'), "w") as f:
                f.create_dataset("features", data=desc, dtype=desc.dtype)
        else:
            desc = db_desc

    # Sanity checks
    if query is None or desc is None:
        raise ValueError("Query or Gallery features failed to load.")

    # Convert to GPU
    output_path = osp.join(args.data_root, dataset, f'nn_{desc_name}.pkl')

    print("Moving features to GPU...")
    query = torch.from_numpy(query).float()
    desc = torch.from_numpy(desc).float()
    
    if torch.cuda.is_available():
        query = query.cuda()
        desc = desc.cuda()
    
    print("Computing Similarity Matrix...")
    scores = query @ desc.T
    
    # Determine offset (0 for query vs gallery, 1 for self-search)
    self_offset = 0
    if dataset in ['gldv2']:
        self_offset = 1
    elif dataset.startswith(('roxford5k', 'rparis6k')):
        m = len(desc)

    scores, ind = torch.topk(scores, k=min(m + self_offset, len(desc)))
    scores = torch.stack((scores, ind)).cpu()[..., self_offset:]

    # Save
    if dataset in ['gldv2', 'sop', 'rp2k', 'ilias']:
        print(f"Saving rankings to {output_path}")
        pickle_save(output_path, scores)
    else:
        pickle_save(output_path, scores)
        class Q:
            def __init__(self):
                self.name = dataset

        out, map, aps = compute_metrics(Q(), scores[1].numpy().T, gnd)


if __name__ == '__main__':
    main()