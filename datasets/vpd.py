# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import random
import math
from pathlib import Path
import numpy as np
import glob
import json
import skimage.io as sio
import matplotlib.pyplot as plt
import os
import skimage.transform
import numpy.linalg as LA
import cv2
import scipy.io as scio

import torch
from torch.utils.data import Dataset

from util.misc import gold_spiral_sampling_patch, change_f


def norm(img, mean=None, std=None):
    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)
    img = (img - mean) / std
    return img


def intersect(a0, a1, b0, b1):
    c0 = ccw(a0, a1, b0)
    c1 = ccw(a0, a1, b1)
    d0 = ccw(b0, b1, a0)
    d1 = ccw(b0, b1, a1)
    if abs(d1 - d0) > abs(c1 - c0):
        return (a0 * d1 - a1 * d0) / (d1 - d0)
    else:
        return (b0 * c1 - b1 * c0) / (c1 - c0)


def ccw(c, a, b):
    a0 = a - c
    b0 = b - c
    return a0[0] * b0[1] - b0[0] * a0[1]


def augment(image, vpts, division):
    if division == 1:  # left-right flip
        return image[:, ::-1].copy(), (vpts * [[-1], [1], [1]]).copy()
    elif division == 2:  # up-down flip
        return image[::-1, :].copy(), (vpts * [[1], [-1], [1]]).copy()
    elif division == 3:  # all flip
        return image[::-1, ::-1].copy(), (vpts * [[-1], [-1], [1]]).copy()
    return image, vpts


def crop(shape, scale=(0.35, 1.0), ratio=(9 / 16, 16 / 9)):
    for attempt in range(20):
        area = shape[0] * shape[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if h <= shape[0] and w <= shape[1]:
            j = random.randint(0, shape[0] - h)
            i = random.randint(0, shape[1] - w)
            return i, j, h, w

    # Fallback
    w = min(shape[0], shape[1])
    i = (shape[1] - w) // 2
    j = (shape[0] - w) // 2
    return i, j, w, w


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split, num_nodes):
        self.rootdir = rootdir
        filelist = sorted(glob.glob(f"{rootdir}/*/*_0.png"))
        print("total number of samples", len(filelist))

        self.split = split
        division = int(len(filelist) * 0.1)
        print("num of valid/test", division)
        if split == "train":
            num_train = int(len(filelist) * 0.8)
            self.filelist = filelist[2 * division: 2 * division + num_train]
            self.size = len(self.filelist)
            print("subset for training: percentage ", 1.0, num_train)
        if split == "val":
            self.filelist = [f for f in filelist[division:division * 2] if "a1" not in f]
            self.size = len(self.filelist)
        if split == "test":
            self.filelist = [f for f in filelist[:division] if "a1" not in f]
            self.size = len(self.filelist)
        print(f"n{split}:", len(self.filelist))

        self.num_nodes = num_nodes
        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90 * np.pi / 180., num_pts=num_nodes)
        """
        0.157: 8.75 degree for 256 anchors;
        0.078: 4.5 degree for 1024 anchors; 
        0.110: 6.3 degree for 512 anchors;
        0.218: 12.5 degree for 128 anchors;
        """
        self.yita = 0.157

    def __len__(self):
        return self.size

    def get_label(self, vpts):
        dis = np.arccos(self.bases @ vpts)
        min_dis = np.min(dis, axis=1)
        conf = -np.ones((self.num_nodes, 1), dtype=np.float32)
        anchor_state_pos = min_dis < self.yita
        anchor_num_pos = anchor_state_pos.sum()
        conf[anchor_state_pos] = 1
        anchor_state_neg = min_dis > 2 * self.yita  # assign the negative instances
        anchor_num_neg = min(2 * anchor_num_pos, anchor_state_neg.sum())
        anchor_index_neg = np.where(anchor_state_neg)
        anchor_sampled_neg = np.random.choice(anchor_index_neg[0], size=anchor_num_neg, replace=False)
        conf[anchor_sampled_neg] = 0

        vps = np.zeros((self.num_nodes, 3), dtype=np.float32)
        vps[anchor_state_pos, :] = vpts.T[np.argmax(self.bases[(conf == 1).squeeze()] @ vpts, 1)]
        return conf, vps

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname).astype(float)[:, :, 0:3]
        # edge_maps = self.get_edge(image, [1./16, 1./8, 1./4])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = norm(image / 255, mean, std)  # RGB
        image = np.rollaxis(image, 2).copy()
        prefix = iname.replace(".png", "")
        with open(f"{prefix}_camera.json") as f:
            js = json.load(f)
            RT = np.array(js["modelview_matrix"])

        vpts = []
        for axis in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]:
            vp = RT @ axis
            vp = np.array([vp[0], -vp[1], -vp[2]])
            vp /= LA.norm(vp)
            if vp[2] < 0.0: vp *= -1.0
            vpts.append(vp)
        vpts = np.array(vpts)
        vpts = vpts.T  # now each column is a vp

        flip = random.random() > 0.5
        if flip & (self.split == "train"):
            image = np.flip(image, axis=2).copy()
            vpts[0, :] = -vpts[0, :]

        # for focal length ablation
        # vpts = change_f(vpts, 2.1875 * 256, 0.75 * 2.1875 * 256)

        conf, vps = self.get_label(vpts)
        r_dict = {'imgs': torch.tensor(image).float(),
                  'vps_unique': torch.tensor(vpts).float(),
                  'vps': torch.tensor(vps).float(),
                  'conf': torch.tensor(conf).float()}

        return r_dict


class ScanNetDataset(Dataset):
    def __init__(self, rootdir, split, num_nodes):
        self.rootdir = rootdir
        self.split = split
        print(self.rootdir, self.split)
        dirs = np.genfromtxt(f"{rootdir}/scannetv2_{split}.txt", dtype=str)
        filelist = sum([sorted(glob.glob(f"{rootdir}/{d}/*.png")) for d in dirs], [])
        print("total number of samples", len(filelist))

        if split == "train":
            filelist = sorted(np.genfromtxt(f"{rootdir}/train.txt", dtype=str))
            self.filelist = filelist
            self.size = len(self.filelist)
        if split == "val":
            random.seed(0)
            random.shuffle(filelist)
            self.filelist = filelist[:2000]
            self.size = len(self.filelist)
        if split == "test":
            random.seed(0)
            random.shuffle(filelist)
            self.filelist = filelist[:2000]  # randomly sample 2k images for a quick test.
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

        self.num_nodes = num_nodes

        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=num_nodes)
        self.yita = 0.157

    def __len__(self):
        return self.size

    def get_label(self, vpts):
        dis = np.arccos(np.abs(self.bases @ vpts))
        min_dis = np.min(dis, axis=1)
        conf = -np.ones((self.num_nodes, 1), dtype=np.float32)
        anchor_state_pos = min_dis < self.yita
        anchor_num_pos = anchor_state_pos.sum()
        conf[anchor_state_pos] = 1
        anchor_state_neg = min_dis > 2 * self.yita  # assign the negative instances
        anchor_num_neg = min(2 * anchor_num_pos, anchor_state_neg.sum())
        anchor_index_neg = np.where(anchor_state_neg)
        anchor_sampled_neg = np.random.choice(anchor_index_neg[0], size=anchor_num_neg, replace=False)
        conf[anchor_sampled_neg] = 0

        vps = np.zeros((self.num_nodes, 3), dtype=np.float32)
        vps[anchor_state_pos, :] = vpts.T[np.argmax(self.bases[(conf == 1).squeeze()] @ vpts, 1)]
        return conf, vps

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname)[:, :, 0:3]
        # edge_maps = self.get_edge(image, [1.])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = norm(image / 255, mean, std)
        image = np.rollaxis(image, 2).copy().astype(float)

        with np.load(iname.replace("color.png", "vanish.npz")) as npz:
            vpts = np.array([npz[d] for d in ["x", "y", "z"]])
            vpts /= LA.norm(vpts, axis=1, keepdims=True)
            vpts = vpts.T

        flip = random.random() > 0.5
        if flip & (self.split == "train"):
            image = np.flip(image, axis=2).copy()
            vpts[0, :] = -vpts[0, :]

        conf, vps = self.get_label(vpts)
        r_dict = {'imgs': torch.tensor(image).float(),
                  'vps_unique': torch.tensor(vpts).float(),
                  'vps': torch.tensor(vps).float(),
                  'conf': torch.tensor(conf).float()}

        return r_dict


class NYUDataset(Dataset):
    def __init__(self, rootdir, split, num_nodes):
        filelist = glob.glob(f"{rootdir}/*.png")
        filelist.sort()
        self.rootdir = rootdir
        self.split = split
        if split == "train":
            self.filelist = filelist[:1000]
            self.size = len(self.filelist) * 4
            print("subset for training: ", self.size)

        if split == "val":
            self.filelist = filelist[1000:1224]
            self.size = len(self.filelist)
            print("subset for valid: ", self.size)

        if split == "test":
            self.filelist = filelist[1224:1449]
            self.size = len(self.filelist)
            print("subset for test: ", self.size)

        if split == "all":
            self.filelist = filelist
            self.size = len(self.filelist)
            print("all: ", len(self.filelist))

        self.num_nodes = num_nodes
        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi / 2, num_pts=num_nodes)
        self.yita = 0.157

    def __len__(self):
        return self.size

    def get_label(self, vpts):
        dis = np.arccos(self.bases @ vpts)
        min_dis = np.min(dis, axis=1)
        conf = -np.ones((self.num_nodes, 1), dtype=np.float32)
        anchor_state_pos = min_dis < self.yita
        anchor_num_pos = anchor_state_pos.sum()
        conf[anchor_state_pos] = 1
        anchor_state_neg = min_dis > 2 * self.yita
        anchor_num_neg = min(2 * anchor_num_pos, anchor_state_neg.sum())
        anchor_index_neg = np.where(anchor_state_neg)
        anchor_sampled_neg = np.random.choice(anchor_index_neg[0], size=anchor_num_neg, replace=False)
        conf[anchor_sampled_neg] = 0

        vps = np.zeros((self.num_nodes, 3), dtype=np.float32)
        vps[anchor_state_pos, :] = vpts.T[np.argmax(self.bases[(conf == 1).squeeze()] @ vpts, 1)]
        return conf, vps

    def __getitem__(self, idx):
        if self.split == "train":
            iname = self.filelist[idx // 4]
        else:
            iname = self.filelist[idx]

        image = skimage.io.imread(iname)[:, :, 0:3]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = norm(image / 255, mean, std)
        image = np.rollaxis(image, 2).copy().astype(float)

        f = 2.4083 * 256
        with np.load(iname.replace(".png", ".npz"), allow_pickle=True) as npz:
            vpts_pixel = npz["vpts_pixel"]
            vps_norm = vpts_pixel - np.array([256, 256])
            vps_norm = np.concatenate((vps_norm, np.ones((vps_norm.shape[0], 1)) * f), axis=1)
            vps_norm /= np.linalg.norm(vps_norm, axis=1, keepdims=True)
        vpts = vps_norm.T  # now each column is a vp

        flip = idx % 4
        if (flip == 1) & (self.split == "train"):
            image = np.flip(image, axis=2).copy()
            vpts[0, :] = -vpts[0, :]
        elif (flip == 2) & (self.split == "train"):
            image = np.flip(image, axis=1).copy()
            vpts[1, :] = -vpts[1, :]
        elif (flip == 3) & (self.split == "train"):
            image = np.flip(image, axis=2).copy()
            image = np.flip(image, axis=1).copy()
            vpts[0:2, :] = -vpts[0:2, :]

        vpts8 = np.zeros((3, 8))
        vpts8[:, :vpts.shape[1]] = vpts

        conf, vps = self.get_label(vpts8)
        r_dict = {'imgs': torch.tensor(image).float(),
                  'vps_unique': torch.tensor(vpts8).float(),
                  'vps': torch.tensor(vps).float(),
                  'conf': torch.tensor(conf).float()}

        return r_dict


class Tmm17Dataset(Dataset):
    def __init__(self, rootdir, split, num_nodes):
        self.rootdir = rootdir
        self.split = split
        if split == "train":
            filelist = np.genfromtxt(f"{rootdir}/train.txt", dtype=str)
        else:
            filelist = np.genfromtxt(f"{rootdir}/val.txt", dtype=str)
        self.filelist = [os.path.join(rootdir, f) for f in filelist]
        if self.split == "train":
            self.size = len(self.filelist) * 4
        else:
            self.size = len(self.filelist)

        print(f"n{split}:", self.size)

        self.num_nodes = num_nodes
        self.bases = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=num_nodes)
        self.yita = 0.157

    def __len__(self):
        return self.size

    def get_label(self, vpts):
        dis = np.arccos(self.bases @ vpts)
        min_dis = np.min(dis, axis=1)
        conf = -np.ones((self.num_nodes, 1), dtype=np.float32)
        anchor_state_pos = min_dis < self.yita
        anchor_num_pos = anchor_state_pos.sum()
        conf[anchor_state_pos] = 1
        anchor_state_neg = min_dis > 2 * self.yita  # assign the negative instances
        anchor_num_neg = min(2 * anchor_num_pos, anchor_state_neg.sum())
        anchor_index_neg = np.where(anchor_state_neg)
        anchor_sampled_neg = np.random.choice(anchor_index_neg[0], size=anchor_num_neg, replace=False)
        conf[anchor_sampled_neg] = 0

        vps = np.zeros((self.num_nodes, 3), dtype=np.float32)
        vps[anchor_state_pos, :] = vpts.T[np.argmax(self.bases[(conf == 1).squeeze()] @ vpts, 1)]
        return conf, vps

    def __getitem__(self, idx):
        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname)
        tname = iname.replace(".jpg", ".txt")
        axy, bxy = np.genfromtxt(tname, skip_header=1)

        a0, a1 = np.array(axy[:2]), np.array(axy[2:])
        b0, b1 = np.array(bxy[:2]), np.array(bxy[2:])
        xy = intersect(a0, a1, b0, b1)

        xy[0] *= 512 / image.shape[1]
        xy[1] *= 512 / image.shape[0]
        image = skimage.transform.resize(image, (512, 512))
        if image.ndim == 2:
            image = image[:, :, None].repeat(3, 2)
        if self.split == "train":
            i, j, h, w = crop(image.shape)
        else:
            i, j, h, w = 0, 0, image.shape[0], image.shape[1]
        image = skimage.transform.resize(image[j : j + h, i : i + w], (512, 512))
        xy[1] = (xy[1] - j) / h * 512
        xy[0] = (xy[0] - i) / w * 512
        f = 2.4083 * 256
        vpts = np.array([[xy[0] - 256, xy[1] - 256, f]])
        vpts[0] /= LA.norm(vpts[0])
        vpts = vpts.T

        image, vpts = augment(image, vpts, idx // len(self.filelist))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = norm(image, mean, std)  # RGB
        image = np.rollaxis(image, 2)

        conf, vps = self.get_label(vpts)
        r_dict = {'imgs': torch.tensor(image).float(),
                  'vps_unique': torch.tensor(vpts).float(),
                  'vps': torch.tensor(vps).float(),
                  'conf': torch.tensor(conf).float()}

        return r_dict


def build(image_set, args):
    if args.dataset_file == "su3":
        args.vpd_path = "/home/dataset/su3"
        dataset = WireframeDataset(args.vpd_path, image_set, args.num_queries)
    if args.dataset_file == "scannet":
        args.vpd_path = "/home/dataset/scannet-vp"
        dataset = ScanNetDataset(args.vpd_path, image_set, args.num_queries)
    if args.dataset_file == "nyu":
        args.vpd_path = "/home/dataset/nyu/nyu_vp"
        dataset = NYUDataset(args.vpd_path, image_set, args.num_queries)
    if args.dataset_file == "tmm17":
        args.vpd_path = "/home/dataset/tmm17"
        dataset = Tmm17Dataset(args.vpd_path, image_set, args.num_queries)
    root = Path(args.vpd_path)
    assert root.exists(), f'provided VPD path {root} does not exist'
    return dataset
