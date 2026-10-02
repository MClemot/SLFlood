import slflood
import slflood.data

import multipers as mp
from multipers.filtrations.density import KDE
import pytest
import torch

# import matplotlib.pyplot as plt

def run(n_pts, n_lms, dim, device):
    pts, lms_idx, function = slflood.data.noisy_sphere_data(n_pts, n_lms, dim, 0.2, seed=0)
    if device in ['cuda', 'triton']:
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        else:
            pts, lms_idx, function = pts.cuda(), lms_idx.cuda(), function.cuda()
    ff = slflood.slflood_bifiltration(pts, lms_idx, function,
                                      use_triton=device=='triton')
    mma_ff = mp.module_approximation(ff)
    # mma_ff.plot(box=[[0, 0], [1, 1]])
    # plt.show()


n_pts = 5000
n_lms = 100


def test_dim2_cpu():
    run(n_pts, n_lms, 2, 'cpu')


def test_dim2_cuda():
    run(n_pts, n_lms, 2, 'cuda')


def test_dim2_triton():
    run(n_pts, n_lms, 2, 'triton')


def test_dim3_cpu():
    run(n_pts, n_lms, 3, 'cpu')


def test_dim3_cuda():
    run(n_pts, n_lms, 3, 'cuda')


def test_dim3_triton():
    run(n_pts, n_lms, 3, 'triton')