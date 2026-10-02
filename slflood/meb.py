import numpy as np
from itertools import combinations

def get_circumspheres(S):
    """
    Computes the circumspheres of a batch of set of points

    Parameters
    ----------
    S : (B, M, N) ndarray, where 1 <= M <= N + 1
            The input points

    Returns
    -------
    C, r : ((2) ndarray, float)
            The center and the squared radius of the circumsphere
    """

    U = S[:,1:] - S[:,0,None]
    B = np.linalg.norm(U, axis=2, keepdims=True)
    U /= B
    B /= 2
    G = np.matmul(U, U.transpose(0,2,1))
    try:
        x = np.linalg.solve(G, B)[:,:,0]
    except np.linalg.LinAlgError:
        G += np.random.normal(size=G.shape, loc=0, scale=1e-10)
        x = np.linalg.solve(G, B)[:,:,0]
    C = np.einsum('bi,bij->bj', x, U)
    r = np.linalg.norm(C, axis=1)
    C += S[:,0]
    return C, r


def get_middles(p):
    """
    Computes the middles of a batch of pair of points
    """
    center = 0.5 * (p[:, 0] + p[:, 1])
    radius = 0.5 * np.linalg.norm(p[:, 1] - p[:, 0], axis=-1)
    return center, radius


def contains_all(center, radius, pts, tol=1e-7):
    """
    Check if ball contains all 4 tetrahedron vertices.
    """
    d = np.linalg.norm(pts - center[:, None, :], axis=-1)
    return np.all(d <= radius[:, None] + tol, axis=-1)


def minimum_enclosing_balls(X:np.ndarray):
    """
    Compute the minimum enclosing balls of a batch of sets of points

    Parameters
    ----------
    X : (B, N, D)
        input

    Returns
    -------
    centers,radius : ((2) ndarray, float)
        centers and radii of the minimum enclosing balls
    """
    B = X.shape[0]
    N = X.shape[1]
    D = X.shape[2]

    best_r = np.full(B, np.inf)
    best_c = np.zeros((B, D))

    for k in range(N,1,-1):
        for idx in combinations(range(N), k):
            subset = X[:, idx]

            if k == 2:
                c, r = get_middles(subset)
            else:
                c, r = get_circumspheres(subset)

            valid = contains_all(c, r, X)
            update = valid & (r < best_r)

            best_r[update] = r[update]
            best_c[update] = c[update]

    return best_c, best_r