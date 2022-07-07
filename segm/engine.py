import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import imageio

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
from typing import Optional


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ) -> None:
        self.weight = weight
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        # print(target_one_hot.shape)
        # compute the actual dice score
        loss = 0.
        dims = (1, 2, 3)

        for c in range(len(self.weight)):
            intersection = torch.sum(input_soft[:, c:c + 1] * target_one_hot[:, c:c + 1], dims)
            cardinality = torch.sum(input_soft[:, c:c + 1] + target_one_hot[:, c:c + 1], dims)

            dice_score = 2. * intersection / (cardinality + self.eps)

            loss += self.weight[c] * torch.mean(1. - dice_score)

        return loss  # /self.weight.sum()


# PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs
        targets = one_hot(targets, num_classes=len(self.weight),
                          device=inputs.device, dtype=inputs.dtype)
        targets = targets

        loss = 0.
        for c in range(len(self.weight)):
            input = inputs[:, c:c + 1].reshape(-1)
            target = targets[:, c:c + 1].reshape(-1)
            intersection = (input * target).sum()
            total = (input + target).sum()
            union = total - intersection

            IoU = (intersection + smooth) / (union + smooth)
            loss += self.weight[c] * (1 - IoU)

        return loss  # /self.weight.sum()


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        epoch,
        amp_autocast,
        loss_scaler,
        weights_cfg
):

    weights = torch.Tensor(weights_cfg).to(ptu.device)
    criterion1 = torch.nn.CrossEntropyLoss(weight=weights)
    criterion2 = DiceLoss(weight=weights)
    criterion3 = IoULoss(weight=weights)

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100
    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)
        with amp_autocast():
            seg_pred = model.forward(im)

            loss1 = criterion1(seg_pred, seg_gt)
            loss2 = criterion2(seg_pred, seg_gt)
            loss3 = criterion3(seg_pred, seg_gt)
            loss = loss1 * 10 + loss2 * 0.5 + loss3 * 0.5
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger


@torch.no_grad()
def evaluate(
        model,
        data_loader,
        val_seg_gt,
        window_size,
        window_stride,
        amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
