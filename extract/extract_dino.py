import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def fix_query_pos(feature_storage, gnd, im_paths):
    im_sizes = np.asarray([i.split(',')[-2:] for i in im_paths]).astype(int)
    loc = feature_storage.storage['local'][..., :2]
    crops = np.asarray([i['bbx'] for i in gnd])[:, None]
    loc = loc * (crops[..., 2:] - crops[..., :2])
    loc = (loc + crops[..., :2]) / im_sizes[:, None]
    feature_storage.storage['local'][..., :2] = loc

def find_divisors(number):
    divisors = np.arange(1, int(np.sqrt(number)) + 1)
    divisors = divisors[number % divisors == 0]
    return divisors


def calculate_receptive_boxes(imsize, ps):
    imsize = torch.tensor(imsize)
    pc_x, pc_y = imsize // ps

    loc = torch.arange(max(pc_x, pc_y)) * ps + (ps / 2)
    loc_x = loc[None, :pc_x]
    loc_y = loc[None, :pc_y]
    boxes = torch.stack([loc_x.repeat_interleave(pc_y, dim=1), loc_y.tile(pc_x)], dim=-1)
    boxes /= imsize
    return boxes

def non_maxima_suppression_2d(heatmap):
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap >= 0.9*hmax)
    return keep

def gem(x, p=3, eps=1e-6):
    return x.clamp(min=eps).pow(p).sum(1).pow(1./p)

def get_local(local_features, local_weights, imsizes, ps, topk=100):
    feature_list, feature_w, scale_limits, boxes, keeps = [], [], [], [], []
    last_scale_limit = 0

    for j, local_feature in enumerate(local_features):
        w = local_weights[j]

        rf_boxes = calculate_receptive_boxes(imsizes[j].shape[-2:], ps)
        boxes.append(rf_boxes)

        local_feature = local_feature.flatten(start_dim=-2).permute(0, 2, 1)
        feature_list.append(local_feature)
        feature_w.append(w.flatten(start_dim=1))

        last_scale_limit += local_feature.shape[1]
        scale_limits.append(last_scale_limit)

    feats = torch.cat(feature_list, dim=1)
    boxes = torch.cat(boxes, dim=1)
    norms = torch.cat(feature_w, dim=1)
    seq_len = min(feats.shape[1], topk)

    weights, ids = torch.topk(norms, k=seq_len, dim=1)
    top_feats = torch.gather(feats, 1, ids[...,None].repeat(1,1,feats.shape[-1]))
    scale_enc = torch.bucketize(ids, torch.asarray(scale_limits).cuda(), right=True)
    locations = torch.gather(boxes.cuda(), 1, ids[...,None].repeat(1,1,2))

    return top_feats, weights, scale_enc, locations, seq_len

@torch.no_grad()
def extract_features(model, detector, test_loader, feature_storage, ps=14, topk=700, chunk_size=10):
    total = 0

    with torch.no_grad():
        img_global_feats, img_cls_global_feats, img_local_feats = [], [], []

        for i, (im_list, scale_list) in enumerate(tqdm(test_loader, mininterval=10)):
            global_features, global_cls_features, local_features, local_weights = [], [], [], []

            for idx in range(len(im_list)):
                im = im_list[idx][:, [2, 1, 0]].cuda()
                cls, feats, weights = model(im)

                div = find_divisors(feats.shape[1])
                feats = feats.permute(0, -1, -2).reshape(feats.shape[0], feats.shape[-1], div[-1], -1)
                if detector:
                    feats, weights = detector(feats)
                else:
                    weights = torch.functional.norm(feats, p=2, dim=1, keepdim=True)

                global_cls_features.append(F.normalize(cls, p=2, dim=-1))
                global_features.append(F.normalize((weights * feats).mean((2,3)), p=2, dim=-1))
                local_features.append(feats)
                local_weights.append(weights)

            top_feats, weights, scale_enc, locations, seq_len = get_local(local_features, local_weights, im_list, ps, topk=topk)

            local_info = torch.zeros((top_feats.shape[0], topk, 773))
            local_info[:, :seq_len] = torch.cat((locations, scale_enc[..., None], torch.zeros_like(weights)[..., None], weights[..., None], top_feats), dim=-1).cpu()
            local_info[:, seq_len:, 3] = 1
            img_local_feats.extend(local_info)
            img_global_feats.extend(F.normalize(torch.stack(global_features, dim=1).mean(1), p=2, dim=-1))
            img_cls_global_feats.extend(F.normalize(torch.stack(global_cls_features, dim=1).mean(1), p=2, dim=-1))
            total += len(im_list[0])

            if total % chunk_size == 0 or total == len(test_loader.dataset):
                if len(img_local_feats) > 0:
                    feature_storage.save(torch.stack(img_cls_global_feats, dim=0), 'cls')
                    feature_storage.save(torch.stack(img_global_feats, dim=0), 'global')
                    feature_storage.save(torch.stack(img_local_feats, dim=0), 'local')
                    feature_storage.update_pointer(len(img_global_feats))
                    img_local_feats, img_cls_global_feats, img_global_feats = [], [], []


def load_dinov2():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

    def dino_forward_hook(module, x, res):
        return res["x_norm_clstoken"], res["x_norm_patchtokens"], None

    model.forward = model.forward_features
    model.register_forward_hook(dino_forward_hook)

    return model


def extract(model, detector, feature_storage, dataloader, topk, im_paths):

    extract_features(model, detector, dataloader, feature_storage, topk=topk)

    if dataloader.dataset.gnd:
        fix_query_pos(feature_storage, dataloader.dataset.gnd, im_paths)
    return feature_storage.storage
