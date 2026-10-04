"""Implementation of sublevel flood bifiltration."""

from typing import Union

import gudhi
import multipers as mp
import numpy as np
import torch

from .landmarks import generate_landmarks_FPS
from .meb import minimum_enclosing_balls
from .utils import generate_grid, mask_from_adjacency, pad_bigrades, reset, elapsed

try:
    from .triton_maxcummin import maxcummin
    from .triton_mask_topk import compact_mask_indices
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

@torch.no_grad()
def near_validity(points: torch.Tensor,
                  simplex_centers: torch.Tensor,
                  simplex_radii: torch.Tensor) -> torch.Tensor:
    centers_to_points = torch.cdist(simplex_centers, points)
    return centers_to_points < simplex_radii.unsqueeze(1)


@torch.no_grad()
def vertex_bigrades(points: torch.Tensor,
                    landmarks: torch.Tensor,
                    function: torch.Tensor,
                    cells: torch.Tensor,
                    mask: torch.Tensor):
    landmarks_to_points = torch.cdist(landmarks, points)
    cummin = torch.cummin(landmarks_to_points, dim=1)[0]
    critical = torch.ones_like(cummin, dtype=torch.int8)
    critical[:, 1:] = cummin[:, 1:] != cummin[:, :-1]

    critical_count = critical.sum(1)
    critical_count_max = critical_count.max().item()
    critical_indices = torch.argsort(critical, dim=1, descending=True, stable=True)[:,:critical_count_max]
    bigrades_diameters = torch.gather(cummin, dim=1, index=critical_indices)
    bigrades_functions = function[critical_indices]
    bigrades = torch.stack((bigrades_diameters, bigrades_functions), dim=-1).cpu().to(torch.float32)

    scc0 = [bigrades[k,:critical_count[k]] for k in range(len(landmarks))]

    cell_mask = critical[cells].amax(dim=1)
    mask = torch.max(mask, cell_mask)

    return scc0, mask


@torch.no_grad()
def slflood_bifiltration(
    points: torch.Tensor,
    landmarks: Union[int, torch.Tensor],
    function: torch.Tensor,
    max_dimension: int = -1,
    points_per_edge: int = 20,
    near_radius: float = np.sqrt(2),
    exact: bool = False,
    batch_size: int = 32,
    use_triton: bool = False,
    fps_h: Union[None, int] = None,
    start_idx: Union[int, None] = 0,
    verbose = False,
):
    """
    Constructs the sublevel Flood bifiltration from a set of points with scalar values and a set of landmarks.

    Args:
        points (torch.Tensor):
            A (N, d) tensor containing the input point set.
        landmarks (Union[int, torch.Tensor]):
            Either an integer indicating the number of landmarks to randomly sample from `points` using FPS, or a tensor of shape (N_l, d) specifying an explicit set of landmarks.
        function (torch.Tensor):
            A (N) tensor containing the scalar values on the points.
        max_dimension (int, optional):
            The top dimension of the simplices to construct.
            Defaults to -1 resulting in the dimension of the ambient space.
        points_per_edge (int, optional):
            Specifies resolution on simplices used for computing filtration values. Defaults to 20.
        near_radius (float, optional):
            Specifies the dilation factor of the minimum enclosing balls to obtain near points. Defaults to sqrt(2).
        exact (bool, optional):
            Whether to compute the exact (unmasked) or approximate (masked) sublevel Flood bifiltration.
        batch_size (int, optional):
            Size of simplex batches. Default to 32.
        use_triton (bool, optional):
            Whether to use Triton kernels if available.
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to
            select the landmarks. If None, then h is selected based on the size of the
            point cloud. Defaults to None.
        start_idx (int | None, optional):
            If provided, FPS starts from this index in the point cloud. If not,
            the start index will be randomly picked from the point cloud. Defaults to 0.

    Returns:
        mp.SimplexTreeMulti
            Returns a k-critical mp.SimplexTreeMulti containing the sublevel Flood bifiltration.
    """

    reset()

    if max_dimension == -1:
        max_dimension = points.shape[1]
    if isinstance(landmarks, int):
        if landmarks >= points.shape[0]:
            landmarks = points
        else:
            landmarks = generate_landmarks_FPS(points, landmarks, fps_h, start_idx=start_idx)
    elif landmarks.device != points.device:
        raise RuntimeError(f"landmarks.device ({landmarks.device}) != points.device ({points.device})")

    device = points.device
    dtype = points.dtype
    use_triton = use_triton and HAS_TRITON and device.type == 'cuda'

    arg = torch.argsort(function)
    ranks = torch.empty_like(arg)
    ranks[arg] = torch.arange(points.size(0), device=device)
    points = points[arg].contiguous()
    function = function[arg].contiguous()

    if not torch.is_floating_point(landmarks):
        landmarks = points[ranks[landmarks]]

    if verbose:
        elapsed("SORT")

    stree = gudhi.delaunay_complex.DelaunayComplex(landmarks.cpu()).create_simplex_tree(filtration=None)
    cells_list = []
    cells_idx  = dict()
    simplices_list = [[] for _ in range(max_dimension-1)]
    simplices_cocells = [[] for _ in range(max_dimension-1)]
    for simplex, _ in stree.get_simplices():
        dim = len(simplex)-1
        if dim == max_dimension:
            cells_idx[tuple(simplex)] = len(cells_list)
            cells_list.append(simplex)
        elif 0 < dim < max_dimension:
            simplices_list[dim-1].append(simplex)
            cocells = []
            for cell,_ in stree.get_cofaces(simplex, max_dimension - dim):
                cocells.append(cells_idx[tuple(cell)])
            simplices_cocells[dim-1].append(cocells)

    ret_stree = mp.SimplexTreeMulti(num_parameters=2, kcritical=True)

    cells = torch.tensor(cells_list, device=device)
    simplices_by_dim = [torch.tensor(l, device=device) for l in simplices_list]

    if verbose:
        elapsed("DELAUNAY")

    # precompute simplex centers
    simplex_vertices = landmarks[cells]
    simplex_vertices_cpu = simplex_vertices.cpu().numpy().astype(np.float64, copy=False)
    Lc, Lr = minimum_enclosing_balls(simplex_vertices_cpu)
    simplex_radii = near_radius * torch.tensor(Lr, device=device, dtype=dtype)
    simplex_centers = torch.tensor(Lc, device=device, dtype=dtype)

    if verbose:
        elapsed("BALLS")

    mask = near_validity(points, simplex_centers, simplex_radii)
    scc0, mask = vertex_bigrades(points, landmarks, function, cells, mask)
    if exact:
        mask = torch.ones_like(mask)
    if verbose:
        elapsed("MASK")

    ret_stree.insert_batch(np.arange(len(scc0))[None,:], pad_bigrades(scc0))
    if verbose:
        elapsed("ASSIGN_0")

    submasks = [mask_from_adjacency(cocells, mask) for cocells in simplices_cocells]
    if verbose:
        elapsed("SUBMASKS")

    simplices_by_dim.append(cells)
    submasks.append(mask)

    for simplices, mask in zip(simplices_by_dim, submasks):
        num_simplices, dim = simplices.shape

        weights = generate_grid(points_per_edge, dim-1, device, dtype)
        simplex_vertices = landmarks[simplices]
        points_on_simplex = weights.unsqueeze(0) @ simplex_vertices

        if not exact:
            mask_counts = mask.sum(1, keepdim=False, dtype=torch.int32)
            indices = torch.sort(mask_counts)[1]
            mask = mask[indices].contiguous()
            simplices = simplices[indices].contiguous()
            points_on_simplex = points_on_simplex[indices].contiguous()

            mask_counts_t = mask_counts[indices]
            mask_counts = mask_counts[indices].cpu().numpy()

        bigrades = []
        for k in range(-(-num_simplices // batch_size)):
            s1, s2 = k*batch_size, min((k+1)*batch_size, num_simplices)

            # b_simplices = simplices[s1:s2]
            # b_simplices_vertices = landmarks[b_simplices]
            # b_sample = weights.unsqueeze(0) @ b_simplices_vertices
            b_sample = points_on_simplex[s1:s2]

            if not exact:
                b_mask = mask[s1:s2]
                b_mask_count_max = int(mask_counts[s2 - 1])
                if use_triton:
                    b_mask_indices = compact_mask_indices(b_mask, mask_counts_t[s1:s2], b_mask_count_max)
                else:
                    b_mask_indices = torch.argsort(b_mask, dim=1, descending=True, stable=True)[:, :b_mask_count_max]
                b_mask_points = points[b_mask_indices]
                b_mask_function = function[b_mask_indices]
            else:
                b_mask_points = points.repeat(s2-s1, 1, 1)
                b_mask_function = function.repeat(s2-s1, 1)

            if use_triton:
                max_diameters = maxcummin(b_mask_points, b_sample)
            else:
                inter = torch.cdist(b_mask_points, b_sample)
                cummin = inter.cummin(dim=1)[0]
                max_diameters = torch.amax(cummin, dim=2)

            critical = torch.ones_like(max_diameters, dtype=torch.int8)
            critical[:, 1:] = max_diameters[:, 1:] != max_diameters[:, :-1]
            critical_count = critical.sum(dim=1)
            critical_indices = torch.argsort(critical, dim=1, descending=True, stable=True)
            bigrade_diameters = torch.gather(max_diameters, dim=1, index=critical_indices)
            bigrade_functions = torch.gather(b_mask_function, dim=1, index=critical_indices)

            b_bigrades = torch.stack((bigrade_diameters, bigrade_functions), dim=-1).cpu().to(torch.float32)
            critical_count = critical_count.cpu().numpy()

            for i in range(len(b_bigrades)):
                bigrades.append(b_bigrades[i, :critical_count[i]].numpy())

        ret_stree.insert_batch(simplices.cpu().numpy().T, pad_bigrades(bigrades))

    if verbose:
        elapsed("LOOP")

    return ret_stree
