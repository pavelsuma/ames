from functools import partial
from typing import List, Tuple, Dict

import hydra
import numpy as np
import torch
import logging
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from models.matcher import AMES
from utils.metrics import mean_average_precision_revisited_rerank
from utils.dataset_loader import get_test_loaders

log = logging.getLogger(__name__)

def evaluate(
        model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        lamb: List[int],
        temp: List[int],
        num_rerank: List[int],
        recall: List[int],
        save_scores: bool = True) -> Tuple[Dict[str, float], float]:
    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
        evaluate_function = partial(mean_average_precision_revisited_rerank, model,
            query_loader, gallery_loader, query_loader.dataset.cache_nn,
            ks=recall,
            lamb=lamb,
            temp=temp,
            top_k=num_rerank,
            gnd=query_loader.dataset.gnd_data,
            save_scores=save_scores
        )
        metrics = evaluate_function()
    return metrics


@hydra.main(config_path="../conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    query_loader, gallery_loader, recall_ks = get_test_loaders(cfg.desc_name, cfg.test_dataset, cfg.num_workers)

    model = AMES(data_root=cfg.data_root, local_dim=gallery_loader.dataset.local_dim, **cfg.model)

    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    model.eval()

    map = evaluate(model=model, lamb=cfg.test_dataset.lamb, temp=cfg.test_dataset.temp, num_rerank=cfg.test_dataset.num_rerank,
                   recall=recall_ks, query_loader=query_loader, gallery_loader=gallery_loader, save_scores=True)
    return map

if __name__ == '__main__':
    main()