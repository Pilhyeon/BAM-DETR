# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from bam_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx, temporal_iou, span_cxw_to_xx_no_clamp

from bam_detr.matcher import build_matcher
from bam_detr.transformer import build_transformer, _get_clones
from bam_detr.position_encoding import build_position_encoding
from bam_detr.misc import accuracy

import numpy as np

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class BAM_DETR(nn.Module):
    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, quality_loss=False,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, is_training=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 3 if span_loss_type == "l1" else max_v_l * 2
        self.span_pred_dim = span_pred_dim
        self.class_embed = nn.Linear(hidden_dim * 3, 2)  # 0: background, 1: foreground

        self.span_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 1, 3) for i in range(span_pred_dim)])

        num_pred = transformer.decoder.num_layers

        self.class_embed = _get_clones(self.class_embed, num_pred)
        self.span_embed = _get_clones(self.span_embed, num_pred)

        for bbox_embed_lst in self.span_embed:
            for bbox_embed in bbox_embed_lst:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        # hack implementation for iterative bounding box refinement
        self.transformer.decoder.bbox_embed = self.span_embed

        self.quality_head = quality_loss
        if self.quality_head:
            self.quality_pred = MLP(hidden_dim * 3, hidden_dim, 1, 3)

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 3)

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.transformer.saliency_proj1 = self.saliency_proj1 
        self.transformer.saliency_proj2 = self.saliency_proj2 

        self.hidden_dim = hidden_dim

        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.global_rep_token_txt = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos_txt = torch.nn.Parameter(torch.randn(hidden_dim))

        self.is_training = is_training

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)

        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        mask_2 = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask, mask_2], dim=1)
        src_2 = self.global_rep_token_txt.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src, src_2], dim=1)
        pos_2 = self.global_rep_pos_txt.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos, pos_2], dim=1)

        video_length = src_vid.shape[1]
        
        self.query_embed.weight.data[..., 1] = torch.minimum(self.query_embed.weight[..., 0], self.query_embed.weight[..., 1])
        self.query_embed.weight.data[..., 2] = torch.minimum(inverse_sigmoid(1. - self.query_embed.weight[..., 0].sigmoid()), self.query_embed.weight[..., 2])
        
        hs, reference, memory, memory_global, memory_boundary = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)

        outputs_class = [self.class_embed[lvl](hs[lvl*2+1]) for lvl in range(self.transformer.decoder.num_layers)]
        outputs_class = torch.stack((outputs_class), dim=0)

        outputs_coord_raw = reference[1:]
        
        left = (outputs_coord_raw[..., 0] - outputs_coord_raw[..., 1])
        right = (outputs_coord_raw[..., 0] + outputs_coord_raw[..., 2])

        outputs_coord = torch.stack(((left + right) / 2.0, right - left), dim=-1)

        left_2 = (outputs_coord_raw[..., 0] - outputs_coord_raw[..., 1].detach())
        right_2 = (outputs_coord_raw[..., 0] + outputs_coord_raw[..., 2].detach())

        outputs_coord2 = torch.stack(((left_2 + right_2) / 2.0, right_2 - left_2), dim=-1)

        if self.quality_head:
            input_feat_center = [hs[2*idx][..., :256] for idx in range(len(reference) - 1)]
            input_feat_boundary = [hs[2*idx + 1][..., 256:] for idx in range(len(reference) - 1)]
            
            input_feat_center = torch.stack(input_feat_center, dim=0)
            input_feat_boundary = torch.stack(input_feat_boundary, dim=0)
            input_feat = torch.cat((input_feat_center, input_feat_boundary - input_feat_center.repeat(1, 1, 1, 2)), dim=-1)

            outputs_quality = self.quality_pred(input_feat).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1], 'memory_boundary': memory_boundary.transpose(0, 1)}

        if self.quality_head:
            out['pred_quality'] = outputs_quality[-1]
        
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg, mask_2], dim=1)
        src_neg = torch.cat([src_, src_neg, src_2], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg, _ = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem

            if self.quality_head:
                out['aux_outputs'] = [
                    {'pred_spans': a, 'pred_spans2': b, 'pred_spans_raw': c, 'pred_quality': d, 'pred_logits': e} for a, b, c, d, e in zip(outputs_coord[:-1], outputs_coord2[:-1], outputs_coord_raw[:-1], outputs_quality[:-1], outputs_class[:-1])]
            else:
                out['aux_outputs'] = [
                    {'pred_spans': a, 'pred_spans2': b, 'pred_spans_raw': c, 'pred_logits': d} for a, b, c, d in zip(outputs_coord[:-1], outputs_coord2[:-1], outputs_coord_raw[:-1], outputs_class[:-1])]
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)

        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(span_cxw_to_xx_no_clamp(src_spans), span_cxw_to_xx(tgt_spans), reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx_no_clamp(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_boundary(self, outputs, targets, indices):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'memory_boundary' in outputs
        
        n_dim = 256

        pred_boundary_left = torch.tanh(outputs['memory_boundary'][..., :n_dim]).mean(dim=-1)
        pred_boundary_right = torch.tanh(outputs['memory_boundary'][..., n_dim:]).mean(dim=-1)

        video_mask = outputs['video_mask']
        targets = targets["span_labels"]

        bs, vid_length = pred_boundary_left.shape

        frame_idxs = torch.arange(vid_length).to(pred_boundary_left.device) + 0.5

        boundary_target_left = torch.zeros_like(pred_boundary_left).to(pred_boundary_left.device)
        boundary_target_right = torch.zeros_like(pred_boundary_right).to(pred_boundary_right.device)

        for b in range(len(targets)):
            span = span_cxw_to_xx(targets[b]['spans']) * vid_length

            for j in range(span.shape[0]):
                left, right = span[j]

                span_len = right - left
                delta_in = 0.1
                delta_out = 0.25

                left_boundary = [left - span_len * delta_out, left + span_len * delta_in]
                right_boundary = [right - span_len * delta_in, right + span_len * delta_out]

                mask_left = (frame_idxs >= left_boundary[0]) & (frame_idxs <= left_boundary[1])
                mask_right = (frame_idxs >= right_boundary[0]) & (frame_idxs <= right_boundary[1])

                boundary_target_left[b][mask_left] = 1
                boundary_target_right[b][mask_right] = 1
                
        weight_boundary_left = boundary_target_left * (1. / (boundary_target_left.sum(dim=-1, keepdim=True) + 1e-9)) + \
                                (1 - boundary_target_left) * (1. / ((1 - boundary_target_left).sum(dim=-1, keepdim=True) + (1 - video_mask).sum(dim=-1, keepdim=True) + 1e-9))
        weight_boundary_right = boundary_target_right * (1. / (boundary_target_right.sum(dim=-1, keepdim=True) + 1e-9)) + \
                                (1 - boundary_target_right) * (1. / ((1 - boundary_target_right).sum(dim=-1, keepdim=True) + (1 - video_mask).sum(dim=-1, keepdim=True) + 1e-9))

        loss_boundary_left = F.binary_cross_entropy(pred_boundary_left, boundary_target_left.detach(), reduction='none')
        loss_boundary_right = F.binary_cross_entropy(pred_boundary_right, boundary_target_right.detach(), reduction='none')
        
        loss_boundary_left = loss_boundary_left * weight_boundary_left
        loss_boundary_right = loss_boundary_right * weight_boundary_right

        loss_boundary = (loss_boundary_left + loss_boundary_right) * 0.5

        loss_boundary = (loss_boundary * video_mask).sum(dim=-1) * 0.5
        loss_boundary = loss_boundary.mean()

        losses = {'loss_boundary': loss_boundary}
        return losses

    def loss_quality(self, outputs, targets, indices):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_spans' in outputs
        assert 'pred_quality' in outputs
        targets = targets["span_labels"]

        src_segments = outputs['pred_spans']        
        pred_quality = outputs['pred_quality']

        bs = src_segments.shape[0]

        gt_iou = []
        for b in range(bs):
            iou_mat, _ = temporal_iou(
                span_cxw_to_xx(src_segments[b].detach()),
                span_cxw_to_xx(targets[b]['spans']))

            iou = iou_mat.max(dim=1)[0]

            gt_iou.append(iou)

        gt_iou = torch.stack(gt_iou, dim=0)
        pred_quality = pred_quality.squeeze(-1)

        gt_iou = gt_iou.flatten()
        pred_quality = pred_quality.flatten()

        pos_ind = torch.nonzero(gt_iou > 0.7)
        m_ind = torch.nonzero((gt_iou <= 0.7) & (
            gt_iou > 0.3))
        neg_ind = torch.nonzero(gt_iou <= 0.3)

        loss_quality = F.l1_loss(pred_quality, gt_iou, reduction='none').float()

        pos_weight = 1 / len(pos_ind) if len(pos_ind) > 0 else 0
        m_weight = 1 / len(m_ind) if len(m_ind) > 0 else 0
        neg_weight = 1 / len(neg_ind) if len(neg_ind) > 0 else 0
        
        loss_quality_pos = torch.sum(loss_quality[pos_ind] * pos_weight).float()
        loss_quality_m = torch.sum(loss_quality[m_ind] * m_weight).float()
        loss_quality_neg = torch.sum(loss_quality[neg_ind] * neg_weight).float()

        loss_quality = (loss_quality_pos + loss_quality_m + loss_quality_neg) / 3.0
        
        losses = {'loss_quality': loss_quality}
        return losses


    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "quality": self.loss_quality,
            "boundary": self.loss_boundary
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        for loss in losses_target:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]   

                for loss in losses_target:
                    if loss in ["saliency", "boundary"]:  # skip as it is only in the top layer
                        continue
                    kwargs = {}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


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


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/bam_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = BAM_DETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            quality_loss=args.quality_loss,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            is_training=False,
        )
    else:
        model = BAM_DETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            quality_loss=args.quality_loss,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            is_training=False,
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.quality_loss:
        weight_dict["loss_quality"] = args.quality_loss_coef
        weight_dict["loss_boundary"] = args.boundary_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.quality_loss:
        losses += ["quality"]
        losses += ["boundary"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
    )
    criterion.to(device)
    return model, criterion
