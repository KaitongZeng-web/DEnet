"""多任务MSE损失函数"""
import torch
import torch.nn as nn


class MultiTaskMSELoss(nn.Module):
    """
    QM9 多任务回归 MSE 损失。
    支持可选 mask：
        mask=True 表示该 target 有效
        mask=False 表示缺失或不参与 loss
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'.")
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target.float()

        # 对齐pred和target形状
        if target.shape != pred.shape:
            if target.numel() == pred.numel():
                target = target.view_as(pred)
            else:
                raise ValueError(
                    f"Target shape {tuple(target.shape)} cannot match pred shape "
                    f"{tuple(pred.shape)}."
                )

        loss = (pred - target) ** 2

        if mask is not None:
            mask = mask.bool()
            loss = loss * mask.float()

        # 损失聚合
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
