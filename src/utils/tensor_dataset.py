import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import pickle_load


class TensorFileDataset(Dataset):
    def __init__(self,
            name: str,
            desc_dir: str,
            local_desc_name: str,
            gnd_data=None,
    ):
        self.name = name
        self.desc_dir = desc_dir
        self.gnd_data = gnd_data
        self.local_file = os.path.join(desc_dir, local_desc_name)
        self.num_desc, self.topk, self.local_dim = h5py.File(self.local_file, 'r')['features'].shape
        self.skip_dim = 5 # skip the first 5 dimensions of the local descriptor (x, y, scale, mask, weight)
        self.local_dim -= self.skip_dim
        
    def __len__(self):
        return self.num_desc


class TestDataset(TensorFileDataset):
    def __init__(self, *args, sequence_len, nn_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_len = sequence_len
        if nn_file is not None:
            nn_inds_path = os.path.join(self.desc_dir, nn_file)
            self.cache_nn = pickle_load(nn_inds_path)

            if self.name == 'gldv2-val':
                self.num_desc = self.num_desc - 750
                self.cache_nn = self.cache_nn[:, :-750]

    def __getitem__(self, batch_index):
        idx = np.sort(np.unique(batch_index)).tolist()
        local_storage = h5py.File(self.local_file, 'r')
        all_local = local_storage['features'][idx, :self.sequence_len]
        local_storage.close()

        masks = all_local[:, :, 3]

        all_local = all_local[[idx.index(i) for i in batch_index]]
        all_local = torch.from_numpy(all_local)

        masks = masks[[idx.index(i) for i in batch_index]]
        masks = torch.from_numpy(masks).bool()

        return (all_local[..., self.skip_dim:], masks), batch_index