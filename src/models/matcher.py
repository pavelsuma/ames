import torch
import torch.nn as nn
import torch.nn.functional as F

from .binarization_layer import BinarizationLayer
from .transformer_dec import TransformerDecoder, TransformerDecoderLayer


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


class AMES(nn.Module):
    def __init__(self, data_root, local_dim, model_dim, nhead, num_encoder_layers, dim_feedforward, activation,
                 normalize_before, binarized=True):
        super(AMES, self).__init__()
        self.binarized = binarized

        self.mtc_token = nn.Parameter(torch.rand(model_dim))
        if self.binarized:
            self.remap_local = nn.Sequential(
                BinarizationLayer(file_name=f'{data_root}/networks/itq_dinov2_D128.npz', trainable=True),
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim))
        else:
            self.remap_local = nn.Sequential(nn.Linear(local_dim, model_dim), nn.LayerNorm(model_dim))

        decoder_layer = TransformerDecoderLayer(model_dim, nhead, dim_feedforward, activation, normalize_before)
        decoder_norm = nn.LayerNorm(model_dim) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers, decoder_norm)
        self.classifier = nn.Linear(model_dim, 1)

        nn.init.trunc_normal_(self.mtc_token, std=.02)
        self.apply(_init_weights)

    def forward(self,
            src_local=None, src_mask=None,
            tgt_local=None, tgt_mask=None,
            return_logits=False):

        src_local = F.normalize(src_local, p=2, dim=-1)
        tgt_local = F.normalize(tgt_local, p=2, dim=-1)

        src_local = self.remap_local(src_local)
        tgt_local = self.remap_local(tgt_local)

        B, Q, D = src_local.size()
        B, T, D = tgt_local.size()

        mtc_token = self.mtc_token[None, None, :].repeat(B, 1, 1)

        if not self.training and not self.binarized:
            tgt_local = tgt_local.half().float()

        input_feats = torch.cat([mtc_token, src_local, tgt_local], 1).permute(1, 0, 2)

        input_mask = torch.cat([
            src_local.new_zeros((B, 1), dtype=torch.bool),
            src_mask,
            tgt_mask
        ], 1).bool()

        sa_mask = src_local.new_zeros((Q+T+1, Q+T+1), dtype=torch.bool)
        sa_mask[1:Q+1, Q+1:] = 1.
        sa_mask[Q+1:, 1:Q+1] = 1.

        ca_mask = src_local.new_zeros((Q+T+1, Q+T+1), dtype=torch.bool)
        ca_mask[1:Q+1, 1:Q+1] = 1.
        ca_mask[Q+1:, Q+1:] = 1.

        logits = self.decoder(input_feats, sa_mask=sa_mask, ca_mask=ca_mask, src_key_padding_mask=input_mask)
        sim = self.classifier(logits[0]).view(-1)

        if return_logits:
            return sim, logits.permute(1, 0, 2)
        return sim
