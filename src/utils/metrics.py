import itertools as it
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple

from .revisited import compute_metrics
from .utils import pickle_save

log = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg

def rerank(model: nn.Module, query_loader, gallery_loader, gnd, cache_nn: torch.Tensor, device, lamb=(0.,), temp=(0.5,),
           top_k=(100,), ks=(1,5,10), save_scores=False, hard=False):
    nn = cache_nn.clone()
    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    if gnd is not None and 'junk' in gnd[0]:
        for i in range(len(cache_nn[0])):
            if hard:
                junk_ids = gnd[i]['junk'] + gnd[i]['easy']
            else:
                junk_ids = gnd[i]['junk']
            is_junk = np.in1d(cache_nn[1, i], junk_ids)
            nn[:, i] = torch.cat((cache_nn[:, i, ~is_junk], cache_nn[:, i, is_junk]), dim=1)
    nn_sims = nn[0]
    nn_inds = nn[1].long()

    max_k = max(top_k)
    scores = []
    for q_f, i in tqdm(query_loader):
        q_score = []
        gallery_loader.batch_sampler.sampler = nn_inds[i, :max_k].T.tolist()
        for db_f, j in tqdm(gallery_loader):
            current_scores = model(
                *list(map(lambda x: x.to(device, non_blocking=True), q_f)),
                *list(map(lambda x: x.to(device, non_blocking=True), db_f)))
            q_score.append(current_scores.cpu().data)
        scores.append(torch.stack(q_score).T)
    raw_sim = torch.cat(scores)

    out = {}
    for k, l, t in it.product(top_k, lamb, temp):
        log.info(f'Rerank={k}, Lambda={l}, Temp={t}')
        s = 1. / (1. + torch.exp(-t * raw_sim))
        s = l * nn_sims[:, :k] + (1 - l) * s[:, :k]
        closest_dists, indices = torch.sort(s, dim=-1, descending=True)
        closest_indices = torch.gather(nn_inds, -1, indices)
        ranks = deepcopy(nn_inds)
        ranks[:, :k] = deepcopy(closest_indices)
        ranks = ranks.cpu().data.numpy().T
        metrics, score, _ = compute_metrics(query_loader.dataset, gallery_loader.dataset, ranks, gnd, kappas=ks)
        out[(k, l, t)] = score

    pickle_save(f'scores1_{"hard" if hard else "medium"}.pkl', raw_sim)
    if save_scores:
        if 'val' in query_loader.dataset.name:
            k, l, t = max(out, key=out.get)
            with open('best_parameters', 'wt') as fid:
                fid.write(f'test_dataset.alpha=[{l}] test_dataset.temp=[{t}]')

    return metrics, score

@torch.no_grad()
def mean_average_precision_revisited_rerank(model: nn.Module, query_loader, gallery_loader,
                                            cache_nn: torch.Tensor, ks: List[int], lamb: List[int], temp: List[int],
                                            top_k: List[int], gnd, save_scores) -> Tuple[Dict[str, float], float]:

    device = next(model.parameters()).device
    out, map = rerank(model, query_loader, gallery_loader, gnd, cache_nn, device, lamb, temp, top_k, ks, save_scores)

    if query_loader.dataset.name.startswith(('roxford5k', 'rparis6k')):
        h_out, h_map = rerank(model, query_loader, gallery_loader, gnd, cache_nn, device, lamb, temp, top_k, ks, save_scores, hard=True)
        out = {
            'M_map': float(out['M_map']),
            'H_map': float(h_out['H_map']),
        }
        log.info(out['M_map'])
        log.info(out['H_map'])
        map = (map + h_map) / 2

    return out, map
