# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os.path

import numpy as np
import sys
from typing import Iterable
import matplotlib.pyplot as plt

import torch

import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        imgs = data['imgs'].to(device)
        # edges = [t.to(device) for t in data['edges']]
        targets = {'vps': data['vps'].to(device), 'conf': data['conf'].to(device),
                   'vps_unique': data['vps_unique'].to(device)}

        outputs = model((imgs, ))

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # print("loss", [loss_dict[k] for k in loss_dict.keys()])
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    errors = []

    for data in metric_logger.log_every(data_loader, 10, header):
        imgs = data['imgs'].to(device)
        # edges = [t.to(device) for t in data['edges']]
        targets = {'vps': data['vps'].to(device), 'conf': data['conf'].to(device),
                   'vps_unique': data['vps_unique'].to(device)}

        outputs = model((imgs, ))

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        outputs, targets = postprocessors(outputs, targets)
        for output, target in zip(outputs, targets['vps_unique']):
            target = target[:, torch.abs(target).sum(dim=0) > 1e-5]  # for yud, ignore paddings
            error = utils.compute_error(output.cpu().numpy(), target.T.cpu().numpy())
            errors += error

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    errors = np.sort(np.array(errors))
    y = (1 + np.arange(len(errors))) / len(errors)
    aa = [f"{utils.AA(errors, y, th):.3f}" for th in [0.5, 1, 2, 3, 5, 10, 20]]
    print(aa)
    return aa


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, device, args=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    errors = []
    pred_num = 0
    gt_num = 0
    # for visualization
    visualization = False

    for data in metric_logger.log_every(data_loader, 10, header):
        imgs = data['imgs'].to(device)
        targets = {'vps': data['vps'].to(device), 'conf': data['conf'].to(device),
                   'vps_unique': data['vps_unique'].to(device)}

        outputs = model((imgs, ))

        outputs, targets = postprocessors(outputs, targets)
        for i, (output, target) in enumerate(zip(outputs, targets['vps_unique'])):
            target = target[:, torch.abs(target).sum(dim=0) > 1e-5]
            gt_num += target.shape[1]
            pred_num += output.shape[0]
            error = utils.compute_error(output.cpu().numpy(), target.T.cpu().numpy())
            errors += error

    errors = np.sort(np.array(errors))
    if args.error_save_path is not None:
        np.savez(args.error_save_path, err=errors)
    y = (1 + np.arange(len(errors))) / len(errors)
    aa = [f"{utils.AA(errors, y, th):.3f}" for th in [0.5, 1, 2, 3, 5, 10, 20]]
    print(aa)