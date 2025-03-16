# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision


def build_dataset(image_set, args):
    if args.dataset_file in ['su3', 'scannet', 'nyu', 'tmm17']:
        from .vpd import build as build_vpd
        return build_vpd(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
