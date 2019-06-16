# -*- coding: utf-8 -*-


import torch

def calc_mean_std(f, eps=1e-5):
    N, C, H, W = f.size()
    f_view = f.view(N, C, -1)  # N x C x (H*W)
    f_mean = torch.mean(f_view, dim=2).view(N, C, 1, 1)
    f_std = (torch.std(f_view, dim=2) + eps).view(N, C, 1, 1)
    return f_mean, f_std

def adaptive_instance_normalization(content_f, style_f):
    cf_mean, cf_std = calc_mean_std(content_f)
    sf_mean, sf_std = calc_mean_std(style_f)
    c_normalised = (content_f - cf_mean) / cf_std
    return sf_std * c_normalised + sf_mean


def _calc_img_flatten_mean_std(img):
    # img 3 X H x W
    img_flatten = img.view(3, -1)
    img_mean = img_flatten.mean(dim=-1, keepdim=True)
    img_std = img_flatten.std(dim=-1, keepdim=True)
    return img_flatten, img_mean, img_std

def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    mat_sqrt = torch.mm(torch.mm(U, torch.diag(D.pow(0.5))), V.t())
    return mat_sqrt

def coral(source, target):  # style, content 3 X H X W
    source_f, source_mean, source_std = _calc_img_flatten_mean_std(source)
    source_f_norm = (source_f - source_mean) / source_std
    source_f_cov = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)
    
    target_f, target_mean, target_std = _calc_img_flatten_mean_std(target)
    target_f_norm = (target_f - target_mean) / target_std
    target_f_cov = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)
    
    source_f_norm_transfer = torch.mm(_mat_sqrt(target_f_cov), torch.mm(torch.inverse(_mat_sqrt(source_f_cov)), source_f_norm))
    
    source_f_transfer = target_std * source_f_norm_transfer + target_mean
    return source_f_transfer.view(source.size())
    
