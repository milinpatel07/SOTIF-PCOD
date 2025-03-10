import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from .anchor_head_single import AnchorHeadSingle


class AnchorHeadSingleVAR(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, *args, **kwargs):
        super().__init__(model_cfg, input_channels, *args, **kwargs)
        self.conv_var = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        nn.init.constant_(self.conv_var.weight, 0)
        nn.init.constant_(self.conv_var.bias, 0)

    def build_losses(self, losses_cfg):
        clf_loss_name = losses_cfg.get('CLF_LOSS_TYPE', 'SigmoidFocalClassificationLoss')
        self.add_module(
            'cls_loss_func',
            getattr(loss_utils, clf_loss_name)(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        loc_loss_weights = [
            losses_cfg.LOSS_WEIGHTS['loc_l1_weight'],
            losses_cfg.LOSS_WEIGHTS['loc_var_weight']
        ]
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(
                code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'],
                l1_weight=losses_cfg.LOSS_WEIGHTS['loc_l1_weight'],
                var_weight=losses_cfg.LOSS_WEIGHTS['loc_var_weight']
            )
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        features_mean = torch.mean(spatial_features_2d, dim=[2, 3])
        features_max = torch.max(torch.max(spatial_features_2d, dim=3)[0], dim=2)[0]
        features_std = torch.std(spatial_features_2d, dim=[2, 3])
        data_dict['batch_features'] = torch.cat([features_mean, features_max, features_std], dim=1)

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        var_preds = self.conv_var(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        var_preds = var_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['var_preds'] = var_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        FORCE_OUTPUT_ANCHOR_LABELS = False # Used in uncertainty evaluation
        if self.training or FORCE_OUTPUT_ANCHOR_LABELS:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_var_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, 
                box_preds=box_preds, 
                var_preds=var_preds,
                dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_var_preds'] = batch_var_preds
            data_dict['cls_preds_normalized'] = False
            if FORCE_OUTPUT_ANCHOR_LABELS:
                data_dict['batch_cls_targets'] = self.forward_ret_dict['box_cls_labels']
        return data_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        clf_loss_name = self.model_cfg.LOSS_CONFIG.get('CLF_LOSS_TYPE', 'SigmoidFocalClassificationLoss')
        if clf_loss_name == 'SigmoidFocalClassificationLoss':
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            one_hot_targets = one_hot_targets[..., 1:]
        elif clf_loss_name == 'SoftmaxFocalLossV1':
            cls_preds = cls_preds.view(batch_size, -1, self.num_class+1)
            one_hot_targets = one_hot_targets[..., 1:]
        elif clf_loss_name == 'SoftmaxFocalLossV2':
            cls_preds = cls_preds.view(batch_size, -1, self.num_class+1)
            one_hot_targets = torch.roll( one_hot_targets, -1, -1 )
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        var_preds = self.forward_ret_dict['var_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        var_preds = var_preds.view(batch_size, -1,
                                   var_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   var_preds.shape[-1])

        loc_loss_src, l1_loss_src, var_loss_src, var_linear_loss_src, var_angle_loss_src = \
            self.reg_loss_func(box_preds, var_preds, box_reg_targets, anchors, \
                               self.box_coder, take_sin_diff=True, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        l1_loss = l1_loss_src.sum() / batch_size
        var_loss = var_loss_src.sum() / batch_size
        var_linear_loss = var_linear_loss_src.sum() / batch_size
        var_angle_loss = var_angle_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item(),
            'rpn_loss_l1': l1_loss.item(),
            'rpn_loss_var': var_loss.item(),
            'rpn_loss_linear_var': var_linear_loss.item(),
            'rpn_loss_angle_var': var_angle_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict


    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, var_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        batch_var_preds = var_preds.view(batch_size, num_anchors, -1) if not isinstance(var_preds, list) \
            else torch.cat(var_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_var_preds = var_preds

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds, batch_var_preds
