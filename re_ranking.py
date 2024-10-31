from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import math

import utils.cmc as cmc
import time

import argparse

from sklearn import metrics
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import os

def fast_mergesetfeat4(_cfg, X, labels):
    """Run GCR for one iteration."""
    if _cfg.GCR.WITH_GPU:
        device = 'cuda'
    else:
        device = 'cpu'
    FIRST_MERGE = True
    start_time = time.time()
    labels_cam = labels[:, 1]
    unique_labels_cam = np.unique(labels_cam)
    index_dic = {item: [] for item in unique_labels_cam}
    for labels_index, item in enumerate(labels_cam):
        index_dic[item].append(labels_index)

    beta1 = _cfg.GCR.BETA1
    beta2 = _cfg.GCR.BETA2
    lambda1 = _cfg.GCR.LAMBDA1
    lambda2 = _cfg.GCR.LAMBDA2
    scale = _cfg.GCR.SCALE

    if _cfg.GCR.MODE == 'fixA' and FIRST_MERGE:
        X2 = torch.sum(X * X, dim=1, keepdim=True)
        r2 = -2. * X @ X.T + X2 + X2.T
        dist = torch.clamp(r2, min=0.)
        FIRST_MERGE = False
    elif _cfg.GCR.MODE == 'fixA' and not FIRST_MERGE:
        dist = torch.load('temp.pt')
    else:
        sim = X @ X.T
        X2 = torch.sum(X * X, dim=1, keepdim=True)
        r2 = -2. * X @ X.T + X2 + X2.T
        dist = torch.clamp(r2, min=0.)

    dis = dist.clone()
    min_value = torch.max(torch.diag(dis))
    min_value = min_value.item()
    means = torch.mean(dis, dim=1, keepdim=True)
    stds = torch.std(dis, dim=1, keepdim=True)
    threshold = means - lambda1 * stds
    threshold = torch.clamp(threshold, min=min_value)
    dis[dis > threshold] = float('inf')

    S = torch.exp(-1*dis / beta1)
    if _cfg.GCR.MODE == 'sym':
        S = 0.5 * (S + S.T)
    D_row = torch.sqrt(1. / torch.sum(S, dim=1))
    D_col = torch.sqrt(1. / torch.sum(S, dim=0))
    L = torch.outer(D_row, D_col) * S
    X = L @ X
        
    if _cfg.GCR.MODE != 'no-norm':
        X = X / torch.linalg.norm(X, ord=2, dim=1, keepdims=True)
    if _cfg.COMMON.VERBOSE:
        print(f'round time {time.time() - start_time} s')
    return X

def fast_run_gcr(_cfg, all_data):
    """Run GCR."""

    [prb_feats, prb_labels, _, gal_feats, gal_labels, _] = all_data
    prb_n = len(prb_labels)
    data = torch.cat([prb_feats, gal_feats], dim=0)
    labels = np.concatenate((prb_labels, gal_labels))
    labels = labels.reshape(labels.shape[0], 1)
    labels = np.repeat(labels, 2, axis=1)
    if _cfg.GCR.WITH_GPU:
        device = 'cuda'
    else:
        device = 'cpu'
    data = data.to(device)
    if _cfg.GCR.ENABLE_GCR:
        for gal_round in range(_cfg.GCR.GAL_ROUND):
            data = fast_mergesetfeat4(_cfg, data, labels)
    prb_feats_new = data[:prb_n, :].cpu()
    gal_feats_new = data[prb_n:, :].cpu()
    return prb_feats_new, gal_feats_new


def fast_gcrv_image(_cfg, all_data):
    [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks] = all_data
    all_data = [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks]
    prb_feats, gal_feats = fast_run_gcr(_cfg, all_data)
    sims = cmc.ComputeEuclid2(prb_feats, gal_feats, 1)
    return sims, prb_feats, gal_feats
