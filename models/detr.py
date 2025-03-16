# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from util import box_ops
from util.misc import gold_spiral_sampling_patch

from .backbone import build_backbone
from .position_encoding import build_position_encoding

from .transformer import build_transformer
from .deformable_transformer import build_deformable_transformer


class VPTR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, pe, transformer, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        if transformer.__class__.__name__ == "Transformer":
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        elif transformer.__class__.__name__ == "DeformableTransformer":
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)  # *2 for d_detr

        input_proj_list = []
        for c in backbone.num_channels:
            input_proj_list.append(nn.Conv2d(c, hidden_dim, kernel_size=1))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.pe = pe

    def forward(self, input):
        features = self.backbone(input[0])
        srcs = []
        for i, f in enumerate(features):
            srcs.append(self.input_proj[i](f))

        pos = []
        for src in srcs:
            pos.append(self.pe(src))
        if self.transformer.__class__.__name__ == "Transformer":
            hs = self.transformer(srcs[0], self.query_embed.weight, pos[0])[0]  # detr
        elif self.transformer.__class__.__name__ == "DeformableTransformer":
            hs = self.transformer(srcs, self.query_embed.weight, pos)[0]  # deformable_detr
        outputs_class = self.class_embed(hs)
        outputs_coord = F.normalize(self.bbox_embed(hs), p=2, dim=-1)
        outputs_coord = outputs_coord * ((outputs_coord[..., -1:] > 0) * 2 - 1)  # for euclidean distance

        out = {'pred_logits': outputs_class[-1], 'pred_pos': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_pos': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class VpLoss(nn.Module):
    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict
        if 'loss_orth' in self.weight_dict.keys():
            self.postprocessors = PostProcess("train")
        # same with dataset
        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=256)
        self.bases = torch.from_numpy(self.bases).float().cuda()
        self.yita = 0.157

    def _get_conf_loss(self, conf_, gt_conf_):
        # only select + and - to calculate loss
        index = gt_conf_ != -1
        conf = conf_[index]
        gt_conf = gt_conf_[index]
        loss_conf = F.binary_cross_entropy_with_logits(conf, gt_conf, reduction='mean')
        return loss_conf

    def _get_pos_loss(self, pos_, gt_pos_, gt_conf):
        # only select + to calculate loss
        index = gt_conf == 1
        pos = pos_[index]
        gt_pos = gt_pos_[index]
        return torch.mean(F.pairwise_distance(pos, gt_pos, p=2))

    def _get_orth_loss(self, outputs, targets):
        outputs, _ = self.postprocessors(outputs, targets)
        loss = 0.
        for output in outputs:
            m = output @ output.T
            mi = torch.eye(m.shape[0], m.shape[1], device=m.device)
            loss += torch.norm(m-mi, p=2)
        return loss / len(outputs)

    def _get_captivity_loss(self, pos_):
        # all for loss
        dis = F.pairwise_distance(pos_, self.bases, p=2) - self.yita
        return dis[dis > 0].sum() / dis.numel()

    def forward(self, outputs, targets):
        conf = outputs['pred_logits'].squeeze()
        gt_conf = targets['conf'].squeeze()
        pos = outputs['pred_pos']
        gt_pos = targets['vps']
        loss = {}
        if 'loss_orth' in self.weight_dict.keys():
            loss.update({'loss_orth': self._get_orth_loss(outputs, targets)})  # for manhattan world
        loss.update({'loss_bce': self._get_conf_loss(conf, gt_conf)})
        loss.update({'loss_pos': self._get_pos_loss(pos, gt_pos, gt_conf)})
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                conf = aux_outputs['pred_logits'].squeeze()
                pos = aux_outputs['pred_pos']
                l_dict = {'loss_bce': self._get_conf_loss(conf, gt_conf),
                          'loss_pos': self._get_pos_loss(pos, gt_pos, gt_conf)}
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                loss.update(l_dict)
        return loss


class PostProcess(nn.Module):
    def __init__(self, phase="val"):
        self.phase = phase
        # same with dataset
        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=256)
        self.bases = torch.from_numpy(self.bases).float().cuda()
        self.yita = 0.157
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, target):
        vpts = target['vps_unique']
        pred_conf = outputs['pred_logits']
        pred_pos = outputs['pred_pos']
        outs = []
        for vpt, p_conf, p_pos in zip(vpts, pred_conf, pred_pos):
            vpt = vpt[:, torch.abs(vpt).sum(dim=0) > 1e-5]  # for yud, ignore vanishing point paddings
            num = vpt.shape[1]
            if self.phase == "train":  # training, cal loss;
                u = (p_conf > 0).squeeze()
                if torch.sum(u) >= num:
                    p_pos = p_pos[u]
                    p_conf = p_conf[u]
            index = torch.argsort(p_conf, dim=0, descending=True)
            candidate = index[0]
            for i in index[1:]:
                if len(candidate) == num:
                    break
                dst = torch.min(torch.arccos(torch.abs(p_pos[candidate] @ p_pos[i].T)))
                if dst < torch.pi / (num + 1):  # for nyu
                    continue
                candidate = torch.cat((candidate, i))
            outs.append(p_pos[candidate])
            # outs.append(self.bases[candidate])  # for debug
        return outs, target


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


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    backbone = build_backbone(args)

    if args.deformable:
        transformer = build_deformable_transformer(args)
    else:
        transformer = build_transformer(args)

    pe = build_position_encoding(args)

    model = VPTR(
        backbone,
        pe,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    weight_dict = {'loss_bce': args.bce_loss_coef, 'loss_pos': args.pos_loss_coef}
    if args.use_orth_loss:
        weight_dict.update({'loss_orth': 1.})
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v * args.aux_loss_coef for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['score', 'pos']
    criterion = VpLoss(weight_dict=weight_dict)
    criterion.to(device)
    postprocessors = PostProcess()

    return model, criterion, postprocessors
