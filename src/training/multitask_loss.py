"""Multi-Task Loss Module for Joint Classification and Lesion Localization.

Combines standard CrossEntropyLoss for disease classification with objectness BCE
(with pos_weight imbalance mitigation) and bounded SmoothL1 box regression.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint classification and lesion localization.

    Formula:
        L_total = L_cls + lambda_loc * (L_obj + lambda_box * L_box)

    where:
        L_cls = CrossEntropyLoss(cls_logits, class_targets)
        L_obj = BCEWithLogitsLoss(pos_weight=pos_weight)(raw_obj, target_obj)
        L_box = SmoothL1Loss(sigmoid(raw_box_coords), target_box_coords) on positive cells only
    """

    def __init__(
        self,
        lambda_loc: float = 0.25,
        lambda_box: float = 1.0,
        pos_weight: Optional[float] = None,
        class_weights: Optional[torch.Tensor] = None,
        smooth_l1_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_loc = lambda_loc
        self.lambda_box = lambda_box
        self.pos_weight = pos_weight

        # Classification loss
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Objectness loss with pos_weight handling
        if pos_weight is not None and pos_weight > 0.0:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pw_tensor", pw_tensor)
            self.obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pw_tensor)
        else:
            self.pw_tensor = None
            self.obj_loss_fn = nn.BCEWithLogitsLoss()

        # Bounding box regression loss
        self.box_loss_fn = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction="mean")

    def forward(
        self,
        cls_logits: torch.Tensor,
        loc_output: torch.Tensor,
        class_targets: torch.Tensor,
        grid_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite multi-task loss.

        Args:
            cls_logits: Predicted class logits [B, num_classes].
            loc_output: Predicted localization grid [B, 5, S, S].
            class_targets: Ground truth class labels [B].
            grid_targets: Ground truth localization grid [B, 5, S, S].

        Returns:
            total_loss: Scalar backpropagatable loss tensor.
            loss_dict: Dictionary containing detached individual loss components.
        """
        device = cls_logits.device

        # 1. Classification Loss
        loss_cls = self.cls_loss_fn(cls_logits, class_targets)

        # 2. Objectness Loss (Channel 0)
        raw_obj = loc_output[:, 0]  # [B, S, S]
        target_obj = grid_targets[:, 0].to(device)  # [B, S, S]

        # Ensure pos_weight is on correct device if buffer exists
        if self.pw_tensor is not None and self.obj_loss_fn.pos_weight.device != device:
            self.obj_loss_fn.pos_weight = self.pw_tensor.to(device)

        loss_obj = self.obj_loss_fn(raw_obj, target_obj)

        # 3. Bounding Box Loss (Channels 1-4, masked on positive cells only)
        # target_obj is binary {0, 1}
        pos_mask = (target_obj > 0.5)  # [B, S, S]

        if pos_mask.sum() > 0:
            # Apply sigmoid to raw box parameters to enforce bounds [0, 1]
            raw_boxes = loc_output[:, 1:5]  # [B, 4, S, S]
            bounded_boxes = torch.sigmoid(raw_boxes)  # [B, 4, S, S]
            target_boxes = grid_targets[:, 1:5].to(device)  # [B, 4, S, S]

            # Expand pos_mask to match 4 box channels: [B, 4, S, S]
            pos_mask_4d = pos_mask.unsqueeze(1).expand_as(bounded_boxes)

            pred_pos = bounded_boxes[pos_mask_4d]
            target_pos = target_boxes[pos_mask_4d]

            loss_box = self.box_loss_fn(pred_pos, target_pos)
        else:
            loss_box = torch.tensor(0.0, device=device)

        # 4. Composite Loss
        loss_loc = loss_obj + (self.lambda_box * loss_box)
        total_loss = loss_cls + (self.lambda_loc * loss_loc)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_loc": float(loss_loc.item()),
            "loss_obj": float(loss_obj.item()),
            "loss_box": float(loss_box.item()),
        }

        return total_loss, loss_dict
