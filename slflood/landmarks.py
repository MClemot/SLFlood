import fpsample
import numpy as np
import torch
from typing import Union

@torch.no_grad()
def generate_landmarks_FPS(
    points: torch.Tensor,
    n_lms: int,
    fps_h: Union[None, int] = None,
    start_idx: Union[int, None] = None,
    return_idxs: bool = False,
) -> torch.Tensor:
    """
    Selects landmarks using Farthest-Point Sampling (bucket FPS).

    This method implements a variant of Farthest-Point Sampling from
    [here](https://dl.acm.org/doi/abs/10.1109/TCAD.2023.3274922).

    Adapted from https://github.com/plus-rkwitt/flooder.

    Args:
        points (torch.Tensor):
            A (P, d) tensor representing a point cloud. The tensor may reside on any
            device (CPU or GPU) and be of any floating-point dtype.
        n_lms (int):
            The number of landmarks to sample (must be <= P and > 0).
        fps_h (Union[None, int], optional):
            h parameter (depth of kdtree) that is used for farthest point sampling to
            select the landmarks. If None, then h is selected based on the size of the
            point cloud. Defaults to None.
        start_idx (int | None, optional):
            If provided, the sampling starts from this index in the point cloud. If not,
            the start index will be randomly picked from the point cloud.
        return_idxs (bool, optional):
            If true, returns the index of the selected landmarks. Defaults to False.

    Returns:
        torch.Tensor:
            A (n_l, d) tensor containing a subset of the input `points`, representing the sampled landmarks,
            or a (n_l) tensor of integers containing the indices of the sampled landmarks.
    """
    if n_lms <= 0:
        raise RuntimeError(
            f"Number of landmarks ({n_lms}) must be positive"
        )
    n_pts = len(points)
    n_lms = min(n_lms, n_pts)
    if fps_h is None:
        if n_pts > 200_000:
            fps_h = 9
        elif n_pts > 80_000:
            fps_h = 7
        else:
            fps_h = 5

    index_set = torch.tensor(
        fpsample.bucket_fps_kdline_sampling(points.cpu(), n_lms, h=fps_h, start_idx=start_idx).astype(np.int64),
        device=points.device,
    )
    if return_idxs:
        return index_set
    return points[index_set]