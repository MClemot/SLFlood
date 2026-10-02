import numpy as np
import torch


@torch.no_grad()
def generate_grid(n: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generates a grid of points on the unit simplex based on the number of points per edge.
    Adapted from https://github.com/plus-rkwitt/flooder.

    Args:
        n (int):
            Number of points per edge.
        dim (int):
            Dimension of the simplex.
        device (torch.device):
            Device to create the tensor on.
        dtype (torch.dtype):
            Dtype of the tensor.

    Returns:
        grid (torch.Tensor):
            Tensor of shape (C, dim + 1), containing the grid points (coordinate weights).
    """

    combs = torch.combinations(torch.arange(n + dim - 1, device=torch.device('cpu')), r=dim)
    padded = torch.cat(
        [
            torch.full((combs.shape[0], 1), -1, device=torch.device('cpu')),
            combs,
            torch.full((combs.shape[0], 1), n + dim - 1, device=torch.device('cpu')),
        ],
        dim=1,
    )  # shape [C, dim + 2]
    grid = torch.diff(padded, dim=1) - 1  # shape [C, dim + 1]
    grid_float = torch.empty_like(grid, dtype=dtype)
    torch.divide(grid, n - 1 , out=grid_float)
    return grid_float.to(device)


@torch.no_grad()
def mask_from_adjacency(adjacency, mask):
    """
    adjacency: list of length n_k, adjacency[i] = list of cell indices adjacent to k-simplex i
    mask:      torch.Tensor of shape (n_cells, *rest), binary

    Returns: torch.Tensor of shape (n_k, *rest)
    """
    rest_shape = mask.shape[1:]
    n_k = len(adjacency)

    lengths = torch.tensor([len(a) for a in adjacency], dtype=torch.long, device=mask.device)
    dest_idx = torch.repeat_interleave(torch.arange(n_k, device=mask.device), lengths)
    src_idx = torch.tensor([c for adj in adjacency for c in adj], dtype=torch.long, device=mask.device)

    values = mask[src_idx]  # (n_edges, *rest)

    idx_scatter = dest_idx.view(-1, *([1] * len(rest_shape))).expand_as(values)
    out = torch.zeros((n_k,) + rest_shape, dtype=mask.dtype, device=mask.device)
    out.scatter_reduce_(0, idx_scatter, values, reduce="amax", include_self=True)

    return out


def pad_bigrades(bigrades_list):
    """
    bigrades_list: list of N arrays, each of shape (k_i, 2), k_i possibly different.
    Returns: array of shape (N, max_k, 2), padded with (inf, inf).
    """
    N = len(bigrades_list)
    max_k = max(arr.shape[0] for arr in bigrades_list)

    padded = np.full((N, max_k, 2), np.inf, dtype=np.float64)
    for i, arr in enumerate(bigrades_list):
        k = arr.shape[0]
        padded[i, :k, :] = arr

    return padded


from colorama import Fore
import time

t = time.time()

def elapsed(str=None):
    global t
    if str is not None:
        print(Fore.MAGENTA + "[timing]" + Fore.RESET, str, int(1000*(time.time() - t)))
    else:
        print(Fore.MAGENTA + "[timing]" + Fore.RESET, int(1000*(time.time() - t)))
    t = time.time()

def reset():
    global t
    t = time.time()