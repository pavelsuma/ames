import os.path as osp
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from .tensor_dataset import TestDataset
from .utils import pickle_load


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def basic_collate(batch):
    return batch[0]


def get_test_sets(desc_name, test_dataset):
    test_gnd_data = None if test_dataset.test_gnd_file is None else pickle_load(osp.join(test_dataset.test_data_dir, test_dataset.test_gnd_file))['gnd']

    gallery_set = TestDataset(test_dataset.name, test_dataset.desc_dir, desc_name + '_gallery_local.hdf5',
                                   sequence_len=test_dataset.sequence_len)
    query_set   = TestDataset(test_dataset.name, test_dataset.desc_dir, desc_name + '_query_local.hdf5',
                                    sequence_len=test_dataset.query_sequence_len, gnd_data=test_gnd_data, nn_file=test_dataset.nn_file)

    return query_set, gallery_set


def get_test_loaders(desc_name, test_dataset, num_workers=8):
    query_set, gallery_set = get_test_sets(desc_name, test_dataset)

    query_sampler = BatchSampler(SequentialSampler(query_set), batch_size=test_dataset.batch_size, drop_last=False)
    gallery_sampler = BatchSampler(SequentialSampler(gallery_set), batch_size=test_dataset.batch_size, drop_last=False)

    query_loader = DataLoader(query_set, sampler=query_sampler, batch_size=1, num_workers=num_workers, pin_memory=test_dataset.pin_memory, collate_fn=basic_collate)
    gallery_loader = DataLoader(gallery_set, sampler=gallery_sampler, batch_size=1, num_workers=num_workers, pin_memory=test_dataset.pin_memory, collate_fn=basic_collate)

    return query_loader, gallery_loader, list(test_dataset.recalls)