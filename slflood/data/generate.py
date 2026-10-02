from multipers.filtrations.density import KDE
import torch
from ..landmarks import generate_landmarks_FPS

device = 'cpu'

def generate_ambient_noise(n_pts, dim, radius=1., seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    res = torch.rand((n_pts*2, dim), device=device) * 2 - 1
    keep = torch.linalg.norm(res, dim=-1) <= 1.
    res = res[keep][:n_pts]
    if res.shape[0] == n_pts:
        return res * radius
    return generate_ambient_noise(n_pts, dim, radius=radius)

def generate_sphere(n, dim, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.normal(0, 1, size=(n,dim), device=device)
    X = X / torch.norm(X, dim=1)[:,None]
    X += torch.normal(0, 0.01, size=(n,dim), device=device)
    return X

def generate_noisy_sphere(n_pts, dim, seed=None):
    pts = generate_sphere(n_pts // 2, dim, device, seed=seed)
    pts = torch.cat([pts, generate_ambient_noise(n_pts // 2, dim, seed=seed) * 1.5])
    return pts

def density(pts, bandwidth):
    f = -KDE(bandwidth=bandwidth, return_log=False).fit(pts).score_samples(pts)
    f -= f.min()
    f /= f.max()
    return f

def sub_pts(pts, n_lms, start_idx=0):
    return generate_landmarks_FPS(pts, n_lms, start_idx=start_idx, return_idxs=True)

def noisy_sphere_data(n_pts, n_lms, dim, bandwidth=.1, seed=None):
    pts = generate_noisy_sphere(n_pts, dim=dim, seed=seed)
    fun = density(pts, bandwidth)
    lms_idx = sub_pts(pts, n_lms)
    return pts, lms_idx, fun