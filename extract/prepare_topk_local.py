import argparse
import os
from glob import glob

import h5py
import os.path as osp
import numpy as np

def test_nonzero_features(desc):
    print(desc.shape)
    subset = np.sort(np.random.choice(len(desc), 1000, replace=False))
    desc_subset = desc[subset]
    norms = np.linalg.norm(desc_subset[..., 5:], axis=-1)
    failed_ids = np.where((norms == 0) * (1 - desc_subset[..., 3]))[0]
    if len(failed_ids):
        print(f"amount failed: {len(failed_ids)}")
    else:
        print("OK")

    return failed_ids

def combine(feat_dir, file_name, dim, num_desc, topk, ext='xa'):
    splits = sorted(glob(osp.join(feat_dir, f'{file_name}_{ext}?.hdf5')))
    if len(splits):
        hdf5_file = h5py.File(os.path.join(feat_dir, f'{file_name}.hdf5'), 'w')
        hdf5_dataset = h5py.VirtualLayout(shape=(num_desc, topk, dim), dtype=np.float32)

        k = 0
        for chunk in splits:
            chunk_ext = chunk.split('.')[0][-3:]
            with open(osp.join(feat_dir, chunk_ext)) as fid:
                num_chunk = len(fid.read().splitlines())

            vsource = h5py.VirtualSource(chunk, 'features', shape=(num_chunk, topk, dim))
            hdf5_dataset[k:k+num_chunk] = vsource
            print(f'{chunk_ext} - OK - {k}:{k + num_chunk}')
            k += num_chunk
        hdf5_file.create_virtual_dataset('features', hdf5_dataset, fillvalue=0)
        test_nonzero_features(hdf5_file['features'])
        hdf5_file.close()
    else:
        raise "No splits to combine."


def main():
    parser = argparse.ArgumentParser(description='Merge hdf5 files.')
    parser.add_argument('--dataset', help='Dataset name to load embeddings of.')
    parser.add_argument('--desc_name', default='dinov2', help='Embeddings to load based on name.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--topk', type=int)
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir
    feat_dir = os.path.join(data_dir, dataset)
    desc_name = args.desc_name
    dim = 773 if 'dinov2' in desc_name else 1029
    topk = args.topk
    ext = ''

    if dataset == 'gldv2':
        with open(osp.join(feat_dir, f'train_750k.txt')) as fid:
            db_lines = fid.read().splitlines()
        combine(feat_dir, f'{desc_name}_local{ext}', dim, len(db_lines), topk)

    elif dataset == 'revisitop1m':
        with open(osp.join(feat_dir, 'test_gallery.txt')) as fid:
            db_lines = fid.read().splitlines()
        combine(feat_dir, f'{desc_name}_local{ext}', dim, len(db_lines), topk)


    elif dataset == 'gldv2-test':
        with open(osp.join(feat_dir, 'test_gallery.txt')) as fid:
            db_lines = fid.read().splitlines()
        combine(feat_dir, f'{desc_name}_gallery_local{ext}', dim, len(db_lines), topk)

    elif dataset in ('roxford5k+1m', 'rparis6k+1m'):
        dataset = dataset.split('+')[0]
        with open(osp.join(data_dir, dataset, 'test_gallery.txt')) as fid:
            db_lines = fid.read().splitlines()
        with open(osp.join(data_dir, 'revisitop1m', 'test_gallery.txt')) as fid:
            r1m_lines = fid.read().splitlines()

        hdf5_file = h5py.File(os.path.join(feat_dir, f'{desc_name}_gallery_local{ext}.hdf5'), 'w')
        hdf5_dataset = h5py.VirtualLayout(shape=(len(db_lines) + len(r1m_lines), topk, dim), dtype=np.float32)

        vsource = h5py.VirtualSource(osp.join(data_dir, dataset, f'{desc_name}_gallery_local{ext}.hdf5'),
                                     'features', shape=(len(db_lines), topk, dim))
        hdf5_dataset[:len(db_lines)] = vsource

        vsource = h5py.VirtualSource(osp.join(data_dir, 'revisitop1m', f'{desc_name}_local{ext}.hdf5'),
                                     'features', shape=(len(r1m_lines), topk, dim))
        hdf5_dataset[len(db_lines):] = vsource

        hdf5_file.create_virtual_dataset('features', hdf5_dataset, fillvalue=0)
        hdf5_file.close()


if __name__ == '__main__':
    main()