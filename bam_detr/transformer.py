# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np
from .attention import MultiheadAttention

import einops

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor, only_center=False):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(256, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 256)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    if only_center:
        return pos_x

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=2, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=3,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=True,
                 is_training=True,
                 ):
        super().__init__()

        txt_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        txt_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.txt_encoder = TransformerEncoder(txt_encoder_layer, num_encoder_layers, txt_encoder_norm)

        t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm)


        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        
        boundary_decoder_layer = BoundaryDecoderLayer(d_model, dim_feedforward, nhead, dropout, activation)

        # decoder_norm = nn.LayerNorm(d_model)
        decoder_norm = None
        self.decoder = TransformerDecoder(decoder_layer, boundary_decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, nhead=nhead,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_t_attn=modulate_t_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                          is_training=is_training)

        self._reset_parameters()

        self.bbox_embed = None
        self.saliency_proj1 = None
        self.saliency_proj2 = None

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns

        self.tgt_embed = nn.Embedding(self.num_queries, 3*self.d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        self.is_training = is_training

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # for tvsum, add video_length in argument
    def forward(self, src, mask, query_embed, pos_embed, video_length=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        txt_src = src[:, video_length + 1:]
        txt_mask = mask[:, video_length + 1:]
        txt_pos = pos_embed[:, video_length + 1:]

        txt_src = txt_src.permute(1, 0, 2)  # (L, batch_size, d)
        txt_pos = txt_pos.permute(1, 0, 2)   # (L, batch_size, d)

        txt_output = self.txt_encoder(txt_src, src_key_padding_mask=txt_mask, pos=txt_pos)  # (L, batch_size, d)
        txt_token = txt_output[-1]

        src = src[:, :-1]
        # src = torch.cat((src[:, :video_length + 1], txt_output[:-1].permute(1, 0, 2)), dim=1)
        mask = mask[:, :-1]
        pos_embed = pos_embed[:, :-1]

        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)  # (L, batch_size, d)
        # print('after encoder : ',src.shape)
        src = src[:video_length + 1]
        mask = mask[:, :video_length + 1]
        pos_embed = pos_embed[:video_length + 1]

        src_local = src[1:]
        src_global = src[0]

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        memory_global, memory_local = memory[0], memory[1:]
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]

        tgt = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1)

        hs, references, memory_boundary = self.decoder(tgt, memory_local, src_local,
                                                        memory_key_padding_mask=mask_local,
                                                        pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed)  # (#layers, #queries, batch_size, d)
        
        memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d)
        return hs, references, memory_local, memory_global, memory_boundary


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # for tvsum, add kwargs
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, boundary_decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=3, nhead=8, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 is_training=True,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        boundary_layers = _get_clones(boundary_decoder_layer, num_layers)
        self.boundary_layers = _get_clones(boundary_layers, 2)

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        self.nhead = nhead

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))


        self.query_scale = _get_clones(self.query_scale, 3)

        self.ref_point_head = MLP(d_model*2, d_model*2, d_model, 2)
        self.ref_point_head_boundary = MLP(d_model, d_model, d_model, 2)

        # for DAB-deter
        if bbox_embed_diff_each_layer:
            module = nn.ModuleList([MLP(d_model, d_model, 1, 3) for i in range(3)])
            self.bbox_embed = nn.ModuleList([module for i in range(num_layers)])
        else:
            module = nn.ModuleList([MLP(d_model, d_model, 1, 3) for i in range(3)])
            self.bbox_embed = module

        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed_layer in self.bbox_embed:
                for bbox_embed in bbox_embed_layer:
                    nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                    nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

        self.is_training = is_training

        self.boundary_conv_left = nn.Sequential(
                                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(d_model, d_model, kernel_size=1, padding=0, bias=True),
                                        nn.ReLU(inplace=True)
                                        )
        self.boundary_conv_right = nn.Sequential(
                                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(d_model, d_model, kernel_size=1, padding=0, bias=True),
                                        nn.ReLU(inplace=True)
                                        )

    def forward(self, tgt, memory, src,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output = tgt

        memory_boundary_neck = src

        memory_boundary_left = self.boundary_conv_left(memory_boundary_neck.permute(1, 2, 0)).permute(2, 0, 1)
        memory_boundary_right = self.boundary_conv_right(memory_boundary_neck.permute(1, 2, 0)).permute(2, 0, 1)

        memory_boundary = torch.cat((memory_boundary_left, memory_boundary_right), dim=-1)
        
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()

        ref_points = [reference_points]

        memory_boundary_left_feat = torch.cat((memory, memory_boundary_left), dim=-1)
        memory_boundary_right_feat = torch.cat((memory, memory_boundary_right), dim=-1)

        for layer_id, (layer, boundary_layer) in enumerate(zip(self.layers, self.boundary_layers)):
            obj_center = reference_points[..., :self.query_dim]

            left = (obj_center[..., 0] - obj_center[..., 1]).clip(min=0)
            right = (obj_center[..., 0] + obj_center[..., 2]).clip(max=1)
            center = (left + right) / 2.0
            length = right - left

            obj_center = torch.stack((center, length), dim=-1)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale[0](output[..., :self.d_model])
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            query_sine_embed = gen_sineembed_for_position(obj_center, only_center=True) * pos_transformation

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output[..., :self.d_model]).sigmoid()  # nq, bs, 1

                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

            output_prev = output.clone()


            output1, attn_matrix1 = layer(output[..., :self.d_model], memory, tgt_mask=tgt_mask,
                           memory_mask=None,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0),
                           is_training=self.is_training)
            
            output2 = output_prev[..., self.d_model:2*self.d_model]
            output3 = output_prev[..., 2*self.d_model:]

            output = torch.cat([output1, output2, output3], dim=-1)
            
            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id][0](output[..., :self.d_model])
                    padding = torch.zeros(tmp.shape[:-1] + (2,)).to(tmp.device)

                    tmp = torch.cat([tmp, padding], dim=-1)
                else:
                    tmp = self.bbox_embed[0](output[..., :self.d_model])
                    padding = torch.zeros(tmp.shape[:-1] + (2,)).to(tmp.device)

                    tmp = torch.cat([tmp, padding], dim=-1)

                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points_center = tmp[..., :self.query_dim].sigmoid()

                reference_points = new_reference_points_center.detach()

            if self.return_intermediate:
                if self.norm is not None:
                    norm_output = [self.norm(output[..., idx*self.d_model:(idx+1)*self.d_model]) for idx in range(3)]
                    norm_output = torch.cat(norm_output, dim=-1)
                    intermediate.append(norm_output)
                else:
                    intermediate.append(output)
            
            obj_center = reference_points[..., :self.query_dim]

            left = (obj_center[..., 0] - obj_center[..., 1]).clip(min=0)
            right = (obj_center[..., 0] + obj_center[..., 2]).clip(max=1)

            center = (left + right) / 2.0
            length = right - left

            obj_center = torch.stack((center, length), dim=-1)

            """ For left """
            obj_center_left = obj_center.clone().detach()

            left_center = center - (length * 0.5)
            left_length = length * 0.5

            obj_center_left = torch.stack((left_center, left_length), dim=-1)

            # get sine embedding for the query vector
            query_sine_embed_left = gen_sineembed_for_position(obj_center_left, only_center=True)

            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation_left = 1
                else:
                    pos_transformation_left = self.query_scale[1](output[..., self.d_model:2*self.d_model])
            else:
                pos_transformation_left = self.query_scale.weight[layer_id]

            query_sine_embed_left = gen_sineembed_for_position(obj_center_left, only_center=True) * pos_transformation_left

            """ For right """
            obj_center_right = obj_center.clone().detach()

            right_center = center + (length * 0.5)
            right_length = length * 0.5

            obj_center_right = torch.stack((right_center, right_length), dim=-1)

            # get sine embedding for the query vector
            query_sine_embed_right = gen_sineembed_for_position(obj_center_right, only_center=True)

            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation_right = 1
                else:
                    pos_transformation_right = self.query_scale[2](output[..., 2*self.d_model:])
            else:
                pos_transformation_right = self.query_scale.weight[layer_id]

            query_sine_embed_right = gen_sineembed_for_position(obj_center_right, only_center=True) * pos_transformation_right

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output[..., :self.d_model]).sigmoid()  # nq, bs, 1
                query_sine_embed_left *= (reft_cond[..., 0] / (obj_center_left[..., 1] * 2)).unsqueeze(-1)
                query_sine_embed_right *= (reft_cond[..., 0] / (obj_center_right[..., 1] * 2)).unsqueeze(-1)

            output_prev = output.clone()

            video_length = (~memory_key_padding_mask).sum(dim=-1).float()

            output2 = boundary_layer[0](memory_boundary_left_feat, obj_center_left[..., 0], output[..., self.d_model:2*self.d_model], video_length)
            output3 = boundary_layer[1](memory_boundary_right_feat, obj_center_right[..., 0], output[..., 2*self.d_model:], video_length)

            output1 = output_prev[..., :self.d_model]
            output = torch.cat([output1, output2, output3], dim=-1)
            
            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:

                    tmp = [self.bbox_embed[layer_id][idx](output[..., idx*self.d_model:(idx+1)*self.d_model]) for idx in range(1, 3)]
                    padding = torch.zeros(tmp[0].shape[:-1] + (1,)).to(tmp[0].device)

                    tmp = torch.cat([padding] + tmp, dim=-1)
                else:

                    tmp = [self.bbox_embed[idx](output[..., idx*self.d_model:(idx+1)*self.d_model]) for idx in range(1, 3)]
                    padding = torch.zeros(tmp[0].shape[:-1] + (1,)).to(tmp[0].device)

                    tmp = torch.cat([padding] + tmp, dim=-1)

                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                
                center_points = new_reference_points_center[..., 0].unsqueeze(-1)
                boundary_points = new_reference_points[..., 1:]

                ref_points.append(torch.cat([center_points, boundary_points], dim=-1))
                
                reference_points = new_reference_points.detach()
            
            if self.return_intermediate:
                if self.norm is not None:
                    norm_output = [self.norm(output[..., idx*self.d_model:(idx+1)*self.d_model]) for idx in range(3)]
                    norm_output = torch.cat(norm_output, dim=-1)
                    intermediate.append(norm_output)
                else:
                    intermediate.append(output)

        if self.norm is not None:
            norm_output = [self.norm(output[..., idx*self.d_model:(idx+1)*self.d_model]) for idx in range(3)]
            norm_output = torch.cat(norm_output, dim=-1)

            output = norm_output
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    memory_boundary,
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    memory_boundary,
                ]

        return [output.unsqueeze(0), memory_boundary]

    
class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        
        assert video_length is not None
        
        pos_src = self.with_pos_embed(src, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]

        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)

        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        print('before src shape :', src.shape)
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]

        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1,0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        print('after src shape :',src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False,
                is_training=True,
                no_self_attn=False):

        # ========== Begin of Self-Attention =============
        if (not self.rm_self_attn_decoder) and (not no_self_attn):
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q_total = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k_total = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)


        if is_training:
            tgt2 = self.cross_attn(query=q_total,
                                key=k_total,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
            attn_matrix = None
        else:
            tgt2, attn_matrix_total = self.cross_attn(query=q_total,
                                key=k_total,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
            
            cross_attn_partial = MultiheadAttention(n_model, self.nhead, dropout=0, vdim=n_model).to(q.device)

            _, attn_matrix_cont = cross_attn_partial(query=q.view(num_queries, bs, n_model),
                                key=k.view(hw, bs, n_model),
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
            _, attn_matrix_pos = cross_attn_partial(query=query_sine_embed.view(num_queries, bs, n_model),
                                key=k_pos.view(hw, bs, n_model),
                                value=v, attn_mask=None,
                                key_padding_mask=memory_key_padding_mask)
            
            del cross_attn_partial
            attn_matrix = torch.stack([attn_matrix_cont, attn_matrix_pos, attn_matrix_total], dim=1)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_matrix


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def bilinear_sampling(value, sampling_locations):
    # values: N, T, N_heads, Dim 
    # sampling_locations: N, N_query, N_heads, N_level, N_points, 2
    N_, T, n_heads, D_ = value.shape
    _, Lq_, n_heads, L_, P_, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    lid_ = 0
    H_ = 1
    W_ = T
    value_l_ = value.permute(0,2,3,1).reshape(N_*n_heads, 1, D_, H_, W_).repeat(1,Lq_,1,1,1)
    value_l_ = value_l_.flatten(0,1)
    sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
    sampling_grid_l_ = sampling_grid_l_.flatten(0,1).unsqueeze(-3)
    sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
    output = sampling_value_l_
    return output.contiguous()

class BoundaryDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu"):
        super().__init__()

        self.d_model = d_model

        self.boundary_deformation = BoundaryDeformation(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, features, proposal_points, pro_features, window_size):

        features = features.transpose(0, 1)
        proposal_points = proposal_points.transpose(0, 1).unsqueeze(-1)
        pro_features = pro_features.transpose(0, 1)

        N, nr_segments = proposal_points.shape[:2]

        point_features = self.boundary_deformation(pro_features, features, proposal_points, window_size)

        point_features = point_features.view(N, nr_segments, self.d_model).permute(1, 0, 2)

        tgt = pro_features.transpose(0, 1)
        tgt2 = point_features

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class BoundaryDeformation(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.d_model = d_model
        self.num_subpoints = 4

        self.nhead = nhead

        self.point_weights = nn.Linear(d_model, self.num_subpoints * nhead)
        self.point_offsets = nn.Linear(d_model, self.num_subpoints * nhead)

        self.value_proj = nn.Linear(d_model*2, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.point_weights.weight.data, 0.)
        nn.init.constant_(self.point_weights.bias.data, 0.)
        nn.init.constant_(self.point_offsets.weight.data, 0.)

        thetas = torch.arange(1, dtype=torch.float32) * (self.num_subpoints * math.pi / 1)
        grid_init = thetas.cos()[:, None]

        grid_init = grid_init.view(1, 1, 1, 1).repeat(
            1, 1, self.num_subpoints, 1)
        for i in range(self.num_subpoints):
            grid_init[:, :, i, :] *= i + 1

        grid_init = torch.cat([grid_init for _ in range(self.nhead)], dim=2)

        with torch.no_grad():
            self.point_offsets.bias = nn.Parameter(grid_init.view(-1))
        
    def forward(self, pro_features, features, boundary_points, window_size):
        features = self.value_proj(features)

        total_len = features.shape[-2]

        N, nr_segments = boundary_points.shape[:2]

        offsets = self.point_offsets(pro_features)
        offsets = einops.rearrange(offsets, 'b n (m p) -> b n m p', m=self.nhead)

        weights = self.point_weights(pro_features)
        weights = einops.rearrange(weights, 'b n (m p) -> p m (b n)', m=self.nhead).softmax(dim=0)
        weights = weights[None, ..., None].repeat(1, 1, 1, 1, self.d_model // self.nhead)

        sampled_points = (boundary_points[..., :, None, None].repeat(1, 1, 1, self.nhead, self.num_subpoints)) + \
            offsets[:, :, None, :, :].repeat(1, 1, 1, 1, 1) / window_size[:, None, None, None, None]

        sampled_points = sampled_points * (window_size[:, None, None, None, None] / total_len)
        
        sampled_points = torch.clamp(sampled_points, min=0., max=1.)

        sampled_points = einops.rearrange(sampled_points, 'b n pb m p -> (b m) n (pb p)')

        grid = sampled_points.new_zeros(sampled_points.shape + (2,))
        grid[:,:,:,0] = sampled_points

        features = einops.rearrange(features, 'b t (m d) -> (b m) t d', m=self.nhead)

        sampled_features = bilinear_sampling(features.unsqueeze(2), grid.unsqueeze(2).unsqueeze(3))

        sampled_features = einops.rearrange(sampled_features, '(b m n) d pb p -> pb p m (b n) d', b=N, n=nr_segments, m=self.nhead)

        features = torch.sum(sampled_features * weights, dim=1)

        features = einops.rearrange(features, 'pb m bn d -> pb bn (d m)')

        features = self.output_proj(features)

        return features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
        is_training=False,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
