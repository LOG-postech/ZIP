# The codes are from Armen Aghajanyan from facebook, from paper
# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
# https://arxiv.org/abs/2012.13255

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Set

def fast_walsh_hadamard_torched(x, axis=0, normalize=False):
    """
    Performs fast Walsh Hadamard transform
    :param x:
    :param axis:
    :param normalize:
    :return:
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), (
        "For a vector of shape %s, axis must be in [0, %d] but it is %d"
        % (orig_shape, len(orig_shape) - 1, axis)
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(torch.prod(torch.tensor(orig_shape[:axis])))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(torch.prod(torch.tensor(orig_shape[axis + 1 :])))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1)
    BB.requires_grad_(False)

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad_(False)

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_()
    GG.requires_grad_(False)
    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))
    return [BB.to(device), Pi.to(device), GG.to(device), divisor.to(device), LL]


def random_vars(desired_dim, intrinsic_dim, device=0):
    """Returns a random matrix of the desired dimension."""
    R = torch.FloatTensor(desired_dim, intrinsic_dim).normal_(std=0.01).to(device)
    R.requires_grad_(False)
    divisor = torch.norm(R)
    return [R, divisor]


def fastfood_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    if not param_list:
        BB, Pi, GG, divisor, LL = fastfood_vars(DD, device=device)

    else:
        BB, Pi, GG, divisor, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0.0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(BB, dd_pad)

    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, GG)

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)

    ret = torch.div(mul_5[:DD], divisor * np.sqrt(float(DD) / LL))

    return ret


def random_torched(intrinsic_vec, param_list: Tuple[torch.Tensor, int]):
    """Random dense transform"""
    R, divisor = param_list
    result = torch.matmul(R, intrinsic_vec)
    # TODO: for now we are not normalizing with the divisor, to be added later.
    return result 