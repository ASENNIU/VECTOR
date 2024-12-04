import logging

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from scipy.special import softmax


def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims
  
def sim_matrix(text_embeds, vid_embeds_pooled):
    """
        Computes the similarity matrix using pooled video frames

        Output
            sims: num_texts x num_vids
        """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    return text_embeds @ vid_embeds_pooled.T

def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1,0,2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy()  # diagonal

    return compute_metrics(ranks)

def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def _compute_metrics(x):
    x = x.numpy()
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics["R100"] = 100 * float(np.sum(ind < 100)) / len(ind)
    metrics['MedR'] = np.median(ind) + 1
    metrics["MeanR"] = np.mean(ind) + 1
    return metrics

def _compute_dsl_metrics(x):
    x = x.numpy()
    x = softmax(x, axis=0) * x
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics["R100"] = 100 * float(np.sum(ind < 100)) / len(ind)
    metrics['MedR'] = np.median(ind) + 1
    metrics["MeanR"] = np.mean(ind) + 1
    
    return metrics