import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.models.resnet import ResNet, extract_feat_res_pycls

RF = 291.0
STRIDE = 16.0
PADDING = 145.0


def generate_coordinates(h, w):
    '''generate coorinates
    Returns: [h*w, 2] FloatTensor
    '''
    x = torch.floor(torch.arange(0, float(w * h)) / w)
    y = torch.arange(0, float(w)).repeat(h)

    coord = torch.stack([x, y], dim=1)
    return coord


def calculate_receptive_boxes(height, width, rf, stride, padding):
    coordinates = generate_coordinates(height, width)
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes


def non_maxima_suppression_2d(heatmap):
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == hmax)
    return keep


def calculate_keypoint_centers(rf_boxes):
    '''compute feature centers, from receptive field boxes (rf_boxes).
    Args:
        rf_boxes: [N, 4] FloatTensor.
    Returns:
        centers: [N, 2] FloatTensor.
    '''
    xymin = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([0, 1]).cuda())
    xymax = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([2, 3]).cuda())
    return (xymax + xymin) / 2.0


@torch.no_grad()
def extract_feature(model, test_loader):
    with torch.no_grad():
        img_feats = [[] for i in range(3)]

        for i, (im_list, scale_list) in enumerate(tqdm(test_loader, mininterval=10)):
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                desc = model(im_list[idx])[0]
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                desc = F.normalize(desc, p=2, dim=1)
                img_feats[idx].append(desc.detach().cpu())

        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(
            torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
        img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg


def get_local(local_features, local_weights, scales, topk=700):
    feature_list, feature_w, scale_limits, boxes, keeps = [], [], [], [], []
    last_scale_limit = 0

    for j, local_feature in enumerate(local_features):
        w = local_weights[j]

        keep = torch.ones_like(w).bool().squeeze(0).squeeze(0)
        # calculate receptive field boxes.
        rf_boxes = calculate_receptive_boxes(
            height=local_feature.size(2),
            width=local_feature.size(3),
            rf=RF,
            stride=STRIDE,
            padding=PADDING)

        # re-projection back to original image space.
        rf_boxes = rf_boxes / torch.stack(scales[j], dim=1).repeat(1, 2)
        boxes.append(rf_boxes.cuda()[keep.flatten().nonzero().squeeze(1)])

        local_feature = local_feature.squeeze(0).permute(1, 2, 0)[keep]
        feature_list.append(local_feature)
        feature_w.append(w.squeeze(0).squeeze(0)[keep])
        last_scale_limit += local_feature.shape[0]
        scale_limits.append(last_scale_limit)

    feats = torch.cat(feature_list, dim=0)
    boxes = torch.cat(boxes, dim=0)
    norms = torch.cat(feature_w, dim=0)
    seq_len = min(feats.shape[0], topk)

    weights, ids = torch.topk(norms, k=seq_len)
    top_feats = feats[ids]
    scale_enc = torch.bucketize(ids, torch.asarray(scale_limits).cuda(), right=True)
    locations = calculate_keypoint_centers(boxes.cuda()[ids])

    return top_feats, weights, scale_enc, locations, seq_len


@torch.no_grad()
def extract_local_feature(model, detector, test_loader, feature_storage, topk=700, chunk_size=5000):
    with torch.no_grad():
        img_feats = []

        for i, (im_list, scale_list) in enumerate(tqdm(test_loader, mininterval=10)):
            local_features, local_weights = [], []
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                feats = extract_feat_res_pycls(im_list[idx], model)[0]
                if detector:
                    feats, weights = detector(feats)
                else:
                    weights = torch.linalg.norm(feats, dim=1).unsqueeze(1)
                local_features.append(feats)
                local_weights.append(weights)

            top_feats, weights, scale_enc, locations, seq_len = get_local(local_features, local_weights, scale_list, topk=topk)

            local_info = torch.zeros((topk, 1029))
            local_info[:seq_len] = torch.cat(
                (locations, scale_enc[:, None], torch.zeros_like(weights)[:, None], weights[:, None], top_feats),
                dim=1).cpu()
            local_info[seq_len:, 3] = 1
            img_feats.append(local_info)

            if (i + 1) % chunk_size == 0:
                if len(img_feats) > 0:
                    feature_storage.save(torch.stack(img_feats, dim=0), 'local')
                    feature_storage.update_pointer(len(img_feats))
                    img_feats = []
        if len(img_feats) > 0:
            feature_storage.save(torch.stack(img_feats, dim=0), 'local')


def load_cvnet(weight_path):
    model = ResNet(101, 2048)

    weight = torch.load(weight_path)
    weight_new = {}
    for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
        weight_new[i.replace('encoder_q.', '')] = j

    mis_key = model.load_state_dict(weight_new, strict=False)
    return model


def extract(model, detector, feature_storage, dataloader, topk):
    if 'global' in feature_storage.save_type:
        features = extract_feature(model, dataloader)
        feature_storage.save(torch.from_numpy(features), 'global')
    if 'local' in feature_storage.save_type:
        extract_local_feature(model, detector, dataloader, feature_storage, topk=topk)
