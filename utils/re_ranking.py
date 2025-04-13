# encoding: utf-8
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

from tqdm import tqdm


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
    # print('K1 is {},  beta1 is {}, K2 is {}, beta2 is {}, scale is {}\n'.format(k1, beta1, k2, beta2, scale))

    # compute global feat
    if _cfg.GCR.MODE == 'fixA' and FIRST_MERGE:
        X2 = torch.sum(X * X, dim=1, keepdim=True)  # 使用 * 进行元素级乘法，使用 torch.sum 求和
        r2 = -2. * X @ X.T + X2 + X2.T  # 使用 @ 符号进行矩阵乘法
        dist = torch.clamp(r2, min=0.)  # 使用 torch.clamp 进行裁剪
        FIRST_MERGE = False
    elif _cfg.GCR.MODE == 'fixA' and not FIRST_MERGE:
        dist = torch.load('temp.pt')  # 使用 torch.load 加载张量
    else:
        sim = X @ X.T  # 使用 @ 符号进行矩阵乘法
        X2 = torch.sum(X * X, dim=1, keepdim=True)  # 使用 * 进行元素级乘法，使用 torch.sum 求和
        r2 = -2. * X @ X.T + X2 + X2.T  # 使用 @ 符号进行矩阵乘法
        dist = torch.clamp(r2, min=0.)  # 使用 torch.clamp 进行裁剪

    dis = dist.clone()
    min_value = torch.max(torch.diag(dis))
    min_value = min_value.item()
    means = torch.mean(dis, dim=1, keepdim=True) #计算每行的均值
    stds = torch.std(dis, dim=1, keepdim=True) #计算每行的标准差
    threshold = means - lambda1 * stds #计算
    threshold = torch.clamp(threshold, min=min_value)
    dis[dis > threshold] = float('inf')

    S = torch.exp(-1*dis / beta1)
    # count_non_zero_elements_per_row(S)
    if _cfg.GCR.MODE == 'sym':
        S = 0.5 * (S + S.T)
    temp = torch.sum(S, dim=1)
    # print(temp.min(), temp.max(), temp.shape)
    temp = torch.sum(S, dim=0)
    # print(temp.min(), temp.max(), temp.shape)
    D_row = torch.sqrt(1. / torch.sum(S, dim=1))
    D_col = torch.sqrt(1. / torch.sum(S, dim=0))
    L = torch.outer(D_row, D_col) * S
    global_X = L @ X  # 使用 @ 符号进行矩阵乘法
        
    X = cross_X 
    if _cfg.GCR.MODE != 'no-norm':
        X = X / torch.linalg.norm(X, ord=2, dim=1, keepdims=True)  # 使用 torch.linalg.norm 进行归一化
    if _cfg.COMMON.VERBOSE:
        print(f'round time {time.time() - start_time} s')
    return X

def fast_run_gcr(_cfg, all_data):
    """Run GCR."""

    [prb_feats, prb_labels, _, gal_feats, gal_labels, _] = all_data
    prb_n = len(prb_labels)
    data = torch.cat([prb_feats, gal_feats], dim=0)
    labels = np.concatenate((prb_labels, gal_labels))
    labels = labels.reshape(labels.shape[0], 1)  # 扩展为 (n, 1)
    labels = np.repeat(labels, 2, axis=1)  # 复制扩展为 (n, 2)
    if _cfg.GCR.WITH_GPU:
        device = 'cuda'
    else:
        device = 'cpu'
    data = data.to(device)
    if _cfg.GCR.ENABLE_GCR:
        for gal_round in range(_cfg.GCR.GAL_ROUND):
            # if _cfg.GCR.MODE != 'localk':
            #     data = mergesetfeat4(_cfg, data, labels)
            # else:
            #     data = mergesetfeat4_localk(_cfg, data, labels)
            data = fast_mergesetfeat4(_cfg, data, labels)
    prb_feats_new = data[:prb_n, :].cpu()
    gal_feats_new = data[prb_n:, :].cpu()
    return prb_feats_new, gal_feats_new


def fast_gcrv_image(_cfg, all_data):
    """GCRV image port."""

    [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks] = all_data
    all_data = [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks]
    prb_feats, gal_feats = fast_run_gcr(_cfg, all_data)
    sims = cmc.ComputeEuclid2(prb_feats, gal_feats, 1)
    return sims, prb_feats, gal_feats
