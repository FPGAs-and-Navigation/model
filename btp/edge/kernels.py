"""Fused CPU kernels for the particle filter (Numba).

The NumPy version of the geophysics likelihood makes several passes over (particles x samples)
temporaries per channel: gather four corners, interpolate, subtract, demean, square, sum. Here
each particle's track is processed in one pass with everything in registers, in parallel over
particles.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _bilinear(m, r, c):
    H, W = m.shape
    r = min(max(r - 0.5, 0.0), H - 1.001)
    c = min(max(c - 0.5, 0.0), W - 1.001)
    r0, c0 = int(r), int(c)
    fr, fc = r - r0, c - c0
    return (m[r0, c0] * (1 - fr) * (1 - fc) + m[r0, c0 + 1] * (1 - fr) * fc
            + m[r0 + 1, c0] * fr * (1 - fc) + m[r0 + 1, c0 + 1] * fr * fc)


@njit(parallel=True, cache=True, fastmath=True)
def geo_cost(maps, mask, rows, cols, y, inv_var, min_inside):
    """Offset-free squared profile mismatch per particle and channel.

    maps (K,H,W) float32, mask (H,W) bool, rows/cols (N,L) fractional grid coordinates of each
    particle's track, y (K,L) measured profile, inv_var (K,) 1/sigma^2.
    Returns cost (N,K) scaled to the full track length, and whether the particle's current
    position (last sample) is on the map. Particles with fewer than `min_inside` in-map samples
    get cost 0 (no information).
    """
    N, L = rows.shape
    K, H, W = maps.shape
    cost = np.zeros((N, K), np.float32)
    on_map = np.zeros(N, np.bool_)
    for i in prange(N):
        n_in = 0
        s = np.zeros(K)
        s2 = np.zeros(K)
        for t in range(L):
            r, c = rows[i, t], cols[i, t]
            ri, ci = int(r), int(c)
            if r < 0 or c < 0 or ri >= H or ci >= W or not mask[ri, ci]:
                continue
            n_in += 1
            for k in range(K):
                d = y[k, t] - _bilinear(maps[k], r, c)
                s[k] += d
                s2[k] += d * d
        r, c = rows[i, L - 1], cols[i, L - 1]
        ri, ci = int(r), int(c)
        on_map[i] = r >= 0 and c >= 0 and ri < H and ci < W and mask[ri, ci]
        if n_in >= min_inside:
            for k in range(K):
                cost[i, k] = (s2[k] - s[k] * s[k] / n_in) * inv_var[k] * L / n_in
    return cost, on_map
