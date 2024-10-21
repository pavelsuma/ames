import copy
from typing import Optional
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                sa_mask: Optional[Tensor] = None,
                ca_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None
                ):

        output = src.permute(1, 0, 2)
        for i, layer in enumerate(self.layers):
            output = layer(output, sa_mask=sa_mask, ca_mask=ca_mask,
                           src_key_padding_mask=src_key_padding_mask)
        output = output.permute(1, 0, 2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, activation, normalize_before):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 2 * dim_feedforward if activation == 'geglu' else dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def _sa_block(self, x: Tensor,
                   attn_mask: Optional[Tensor] = None,
                   src_key_padding_mask: Optional[Tensor] = None,
                  ) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask)[0]
        return x

    def _ca_block(self, x: Tensor,
                   attn_mask: Optional[Tensor] = None,
                   src_key_padding_mask: Optional[Tensor] = None,
                  ) -> Tensor:
        x = self.cross_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask)[0]
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x

    def forward(self,
                 x,
                 sa_mask: Optional[Tensor] = None,
                 ca_mask: Optional[Tensor] = None,
                 src_key_padding_mask: Optional[Tensor] = None):
        if self.normalize_before:
            x = x + self._sa_block(self.norm1(x), sa_mask, src_key_padding_mask)
            x = x + self._ca_block(self.norm2(x), ca_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, sa_mask, src_key_padding_mask))
            x = self.norm2(x + self._ca_block(x, ca_mask, src_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
