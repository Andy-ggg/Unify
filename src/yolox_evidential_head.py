import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import YOLOXHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import reduce_mean
from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class EvidentialYOLOXRegHead(YOLOXHead):
    """
    Extends YOLOXHead to incorporate Evidential Learning for both regression and classification (version 2).

    Regression: Outputs 7 channels:
      - The first 4 channels are (dx, dy, dw, dh) for bounding box regression.
      - The last 3 channels are evidential parameters (v, alpha, beta).

    Classification: Outputs 2*num_classes channels:
      - The first half represents regular classification logits (compatible with YOLOX BCE/FocalLoss).
      - The second half consists of evidential log_evidence used for computing Dirichlet-based NLL + KL loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 # Regression uncertainty parameters
                 reg_evd_loss_weight: float = 0.01,
                 reg_beta: float = 0.1, 
                 max_reg_scale: float = 0.5, 
                 annealing_step: int = 1000,
                 kl_beta: float = 1.0,
                 eps: float = 1e-6,
                 # Classification uncertainty parameters (adjustable)
                 cls_evd_loss_weight: float = 0.01,  # Can be increased/decreased
                 cls_kl_beta: float = 0.001,         # KL term coefficient, kept small to avoid excessive penalty
                 cls_prior_alpha: float = 1.0,       # Dirichlet prior alpha
                 **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)

        # Regression
        self.reg_evd_loss_weight = reg_evd_loss_weight
        self.reg_beta = reg_beta
        self.max_reg_scale = max_reg_scale
        self.annealing_step = annealing_step
        self.kl_beta = kl_beta
        self.eps = eps

        # Classification
        self.cls_evd_loss_weight = cls_evd_loss_weight
        self.cls_kl_beta = cls_kl_beta
        self.cls_prior_alpha = cls_prior_alpha

        # global_step used for annealing
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long), persistent=True)

    def _build_predictor(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        1) Classification branch outputs 2*self.num_classes channels:
           - First half corresponds to normal classification logits (compatible with YOLOX)
           - Second half corresponds to evidential log_evidence
        2) Regression branch outputs 7 channels
        3) Objectness branch outputs 1 channel
        """
        # Classification: 2x channels
        conv_cls = nn.Conv2d(self.feat_channels, 2 * self.cls_out_channels, 1)
        # Regression: 7 channels
        conv_reg = nn.Conv2d(self.feat_channels, 7, 1)
        # Objectness: 1 channel
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self) -> None:
        """Initialize weights, same as YOLOXHead."""
        super(YOLOXHead, self).init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, 
                       x: Tensor, 
                       cls_convs: nn.Module,
                       reg_convs: nn.Module, 
                       conv_cls: nn.Module,
                       conv_reg: nn.Module,
                       conv_obj: nn.Module
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for a single feature map.

        Returns: 
            - cls_score:  [B, num_classes, H, W]          (Standard classification logits)
            - bbox_4c:    [B, 4, H, W] (dx, dy, dw, dh)
            - obj_score:  [B, 1, H, W]
            - others:     (Evidential regression evd_3c and classification cls_evd)
        """
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        all_cls = conv_cls(cls_feat)  # [B, 2*num_classes, H, W]
        all_reg = conv_reg(reg_feat)  # [B, 7, H, W]
        obj_score = conv_obj(reg_feat)

        # Split classification:
        # - First num_classes => standard classification logits
        # - Second num_classes => evidential log_evidence
        half_c = self.cls_out_channels
        cls_score   = all_cls[:, :half_c, :, :]
        cls_evd     = all_cls[:, half_c:, :, :]  # [B, num_classes, H, W]

        # Split regression branch
        bbox_4c     = all_reg[:, :4, :, :]
        evd_3c      = all_reg[:, 4:, :, :]  # (v, alpha, beta)

        return cls_score, bbox_4c, obj_score, (evd_3c, cls_evd)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Overrides YOLOXHead.forward to unpack additional evidential outputs.
        """
        outs = super().forward(feats)
        # outs contains 4 lists: [cls_scores, bbox_preds, obj_scores, other_preds]
        # where other_preds[i] = (evd_3c_i, cls_evd_i)
        cls_scores, bbox_preds_7c, obj_scores, others = outs

        # Separate evidential regression and classification
        final_bbox_preds_4c = []
        self.extra_reg_evd_preds = []
        self.extra_cls_evd_preds = []

        for b7, (evd_3c, cls_evd) in zip(bbox_preds_7c, others):
            final_bbox_preds_4c.append(b7[:, :4, :, :])
            self.extra_reg_evd_preds.append(evd_3c)
            self.extra_cls_evd_preds.append(cls_evd)

        return cls_scores, final_bbox_preds_4c, obj_scores

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds_4c: List[Tensor],
                     objectnesses: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        1. Compute YOLOX original loss for classification / regression / objectness.
        2. Compute Evidential regression loss (NIG distribution).
        3. Compute Evidential classification loss (Dirichlet single-label NLL + KL).
        """
        # 1) Original YOLOX loss
        loss_dict = super().loss_by_feat(
            cls_scores, bbox_preds_4c, objectnesses,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )

        # 2) Evidential regression loss
        reg_evd_loss = self._compute_reg_evidential_loss(
            self.extra_reg_evd_preds, 
            bbox_preds_4c,
            batch_gt_instances, 
            batch_img_metas
        )
        loss_dict['loss_bbox'] = loss_dict.get('loss_bbox', 0) + self.reg_evd_loss_weight * reg_evd_loss

        # 3) Evidential classification loss
        cls_evd_loss = self._compute_cls_evidential_loss(
            self.extra_cls_evd_preds,
            cls_scores,
            batch_gt_instances, 
            batch_img_metas
        )
        loss_dict['loss_cls'] = loss_dict.get('loss_cls', 0) + self.cls_evd_loss_weight * cls_evd_loss

        # Update global_step
        self.global_step.add_(1)
        return loss_dict

    # ------------------ Evidential Regression Part ------------------
    def _compute_reg_evidential_loss(self,
                                     evd_preds_3c: List[Tensor],
                                     bbox_preds_4c: List[Tensor],
                                     batch_gt_instances: InstanceList,
                                     batch_img_metas: List[dict]) -> Tensor:
        """
        Compute evidential regression loss based on Normal-Inverse-Gamma (NIG) distribution.

        - `evd_preds_3c`: Evidential regression predictions [v, alpha, beta]
        - `bbox_preds_4c`: Bounding box predictions (dx, dy, dw, dh)
        - `batch_gt_instances`: Ground-truth instances per image
        - `batch_img_metas`: Image metadata

        This loss penalizes incorrect regression predictions while also learning confidence estimates.
        """
        device = bbox_preds_4c[0].device
        total_evd_loss = torch.zeros((), device=device)
        num_imgs = len(batch_gt_instances)

        for img_id in range(num_imgs):
            gt_instances = batch_gt_instances[img_id]
            if len(gt_instances) == 0:
                continue

            flatten_reg_4c = []
            flatten_evd_3c = []
            flatten_priors  = []

            # Process each feature level
            for lvl_idx in range(len(bbox_preds_4c)):
                H, W = bbox_preds_4c[lvl_idx].shape[2], bbox_preds_4c[lvl_idx].shape[3]
                if H == 0 or W == 0:
                    continue

                reg_4c_l = bbox_preds_4c[lvl_idx][img_id].permute(1, 2, 0).reshape(-1, 4)
                evd_3c_l = evd_preds_3c[lvl_idx][img_id].permute(1, 2, 0).reshape(-1, 3)

                priors_lvl = self.prior_generator.single_level_grid_priors(
                    (H, W), lvl_idx, device
                )
                if priors_lvl.numel() == 0:
                    continue

                if priors_lvl.shape[0] != reg_4c_l.shape[0]:
                    continue

                flatten_reg_4c.append(reg_4c_l)
                flatten_evd_3c.append(evd_3c_l)
                flatten_priors.append(priors_lvl)

            if len(flatten_priors) == 0:
                continue

            flatten_reg_4c = torch.cat(flatten_reg_4c, dim=0)
            flatten_evd_3c = torch.cat(flatten_evd_3c, dim=0)
            flatten_priors = torch.cat(flatten_priors, dim=0)

            # Decode bounding boxes
            try:
                decoded_bboxes = self._bbox_decode(flatten_priors, flatten_reg_4c)
            except Exception:
                continue

            assign_result = self.assigner.assign(
                decoded_bboxes,
                None,
                None,
                gt_instances.bboxes,
                gt_instances.labels,
                batch_img_metas[img_id]
            )

            fg_mask = assign_result.gt_inds > 0
            num_pos = fg_mask.sum()
            if num_pos == 0:
                continue

            pos_decoded = decoded_bboxes[fg_mask]
            matched_gt  = gt_instances.bboxes[assign_result.gt_inds[fg_mask] - 1]

            evd_fg = flatten_evd_3c[fg_mask]
            v = evd_fg[:, 0:1].exp()
            alpha = evd_fg[:, 1:2].exp() + 1.0
            beta  = evd_fg[:, 2:3].exp()

            error = pos_decoded - matched_gt
            error_term = error.square().mean(dim=1, keepdim=True)

            nll = self.evd_nll_loss_reg(error_term, alpha, beta)
            kl  = self.evd_kl_loss_reg(alpha, beta, prior_alpha=3.0, prior_beta=1.0)
            anneal_coef = min(1.0, float(self.global_step.item()) / self.annealing_step)

            total_evd_loss += (nll + self.kl_beta * anneal_coef * kl)

        if num_imgs > 0:
            total_evd_loss /= num_imgs
        return total_evd_loss

    def evd_nll_loss_reg(self, 
                         error_term: Tensor,
                         alpha: Tensor,
                         beta: Tensor) -> Tensor:
        """
        Compute negative log-likelihood (NLL) loss for Normal-Inverse-Gamma (NIG) regression.
        """
        nll = 0.5 * math.log(math.pi) - torch.log(beta) \
              + alpha * torch.log(1.0 + error_term / (2.0 * beta))
        return nll.mean()

    def evd_kl_loss_reg(self, 
                        alpha: Tensor, 
                        beta: Tensor, 
                        prior_alpha: float = 3.0,
                        prior_beta: float = 1.0) -> Tensor:
        """
        Compute KL divergence between the learned distribution and a prior NIG distribution.
        """
        device = alpha.device
        alpha_term = alpha * torch.log(beta / prior_beta)
        log_gamma = - torch.lgamma(alpha) + torch.lgamma(torch.tensor(prior_alpha, device=device))
        digamma_term = (prior_alpha - alpha) * torch.digamma(alpha)
        beta_term = (beta - prior_beta) / prior_beta
        kl = alpha_term + log_gamma + digamma_term + beta_term
        return kl.mean()

    # ------------------ Evidential Classification Part ------------------
    def _compute_cls_evidential_loss(self,
                                 cls_evd_preds: List[Tensor],
                                 cls_scores: List[Tensor],
                                 batch_gt_instances: InstanceList,
                                 batch_img_metas: List[dict]) -> Tensor:
        """
        Compute evidential classification loss using Dirichlet distribution.

        - Uses single-label Dirichlet NLL for positive samples.
        - KL divergence between predicted distribution and a uniform prior.

        `cls_evd_preds`: List of predicted log-evidence maps per feature level.
        """
        device = cls_evd_preds[0].device
        total_cls_evd_loss = torch.zeros((), device=device)
        num_imgs = len(batch_gt_instances)

        for img_id in range(num_imgs):
            gt_instances = batch_gt_instances[img_id]
            if len(gt_instances) == 0:
                continue

            flatten_cls_evd = []
            flatten_priors = []

            # Iterate over feature levels
            for lvl_idx in range(len(cls_evd_preds)):
                H, W = cls_evd_preds[lvl_idx].shape[2], cls_evd_preds[lvl_idx].shape[3]
                if H == 0 or W == 0:
                    continue

                evd_l = cls_evd_preds[lvl_idx][img_id].permute(1, 2, 0).reshape(-1, self.num_classes)
                priors_lvl = self.prior_generator.single_level_grid_priors(
                    (H, W), lvl_idx, device
                )

                if priors_lvl.numel() == 0:
                    continue
                if priors_lvl.shape[0] != evd_l.shape[0]:
                    continue

                flatten_cls_evd.append(evd_l)
                flatten_priors.append(priors_lvl)

            if len(flatten_priors) == 0:
                continue

            flatten_cls_evd = torch.cat(flatten_cls_evd, dim=0)
            flatten_priors  = torch.cat(flatten_priors, dim=0)

            # Convert log-evidence to alpha values
            alpha_fg = flatten_cls_evd.exp() + 1.0  # [N, num_classes]
            alpha_0 = alpha_fg.sum(dim=1, keepdim=True)  # [N,1]

            nll = torch.log(alpha_0) - torch.log(alpha_fg.max(dim=1, keepdim=True)[0])
            kl = self.dirichlet_kl_div(alpha_fg, prior_alpha=self.cls_prior_alpha)

            anneal_coef = min(1.0, float(self.global_step.item()) / self.annealing_step)
            total_cls_evd_loss += (nll.mean() + self.cls_kl_beta * anneal_coef * kl.mean())

        if num_imgs > 0:
            total_cls_evd_loss /= num_imgs
        return total_cls_evd_loss

    def dirichlet_kl_div(self, alpha: Tensor, prior_alpha: float = 1.0) -> Tensor:
        """
        Compute KL divergence between Dirichlet(alpha) and a uniform Dirichlet prior.
        """
        S = alpha.sum(dim=1, keepdim=True)
        K = alpha.shape[1]
        t1 = torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        t2 = - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        t3 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1, keepdim=True)
        return (t1 + t2 + t3).squeeze(-1)
