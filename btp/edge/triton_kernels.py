"""Triton kernel for the particle filter's geophysics likelihood (GPU / Jetson path).

One program handles a block of particles. For every sample of the 1.6 km track it computes the
map coordinate from the particle position and the track offset, bilinearly samples the three
maps (magnetic, gD, GDD), and accumulates the sums needed for the offset-free squared mismatch,
all in registers. The (particles x samples) coordinate and residual arrays that the NumPy
version materialises never exist.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _geo_cost_kernel(px_ptr, py_ptr, te_ptr, tn_ptr, y_ptr, maps_ptr, mask_ptr, inv_ptr,
                     cost_ptr, on_ptr, N, H, W, inv_res, min_inside,
                     L: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    pe = tl.load(px_ptr + i, mask=live, other=0.0)
    pn = tl.load(py_ptr + i, mask=live, other=0.0)
    n_in = tl.zeros([BLOCK], dtype=tl.float32)
    s0 = tl.zeros([BLOCK], dtype=tl.float32)
    s1 = tl.zeros([BLOCK], dtype=tl.float32)
    s2 = tl.zeros([BLOCK], dtype=tl.float32)
    q0 = tl.zeros([BLOCK], dtype=tl.float32)
    q1 = tl.zeros([BLOCK], dtype=tl.float32)
    q2 = tl.zeros([BLOCK], dtype=tl.float32)
    last_in = tl.zeros([BLOCK], dtype=tl.int1)
    HW = H * W
    for t in tl.static_range(L):
        # local coordinates: pe = metres east of the map's west edge, pn = metres south of its north edge
        c = (pe + tl.load(te_ptr + t)) * inv_res
        r = (pn - tl.load(tn_ptr + t)) * inv_res
        ri = r.to(tl.int32)
        ci = c.to(tl.int32)
        inb = live & (r >= 0) & (c >= 0) & (ri < H) & (ci < W)
        m = tl.load(mask_ptr + ri * W + ci, mask=inb, other=0) != 0
        inside = inb & m
        # bilinear on cell centres (same convention as geo.bilinear)
        rr = tl.minimum(tl.maximum(r - 0.5, 0.0), H - 1.001)
        cc = tl.minimum(tl.maximum(c - 0.5, 0.0), W - 1.001)
        r0 = rr.to(tl.int32)
        c0 = cc.to(tl.int32)
        fr = rr - r0
        fc = cc - c0
        w00 = (1 - fr) * (1 - fc)
        w01 = (1 - fr) * fc
        w10 = fr * (1 - fc)
        w11 = fr * fc
        base = r0 * W + c0
        f = inside.to(tl.float32)
        n_in += f
        for k in tl.static_range(K):
            mk = maps_ptr + k * HW + base
            v = (tl.load(mk, mask=inside, other=0.0) * w00 + tl.load(mk + 1, mask=inside, other=0.0) * w01
                 + tl.load(mk + W, mask=inside, other=0.0) * w10 + tl.load(mk + W + 1, mask=inside, other=0.0) * w11)
            d = (tl.load(y_ptr + k * L + t) - v) * f
            if k == 0:
                s0 += d
                q0 += d * d
            elif k == 1:
                s1 += d
                q1 += d * d
            else:
                s2 += d
                q2 += d * d
        if t == L - 1:
            last_in = inside
    ok = n_in >= min_inside
    scale = tl.where(ok, L / tl.maximum(n_in, 1.0), 0.0)
    nn = tl.maximum(n_in, 1.0)
    tl.store(cost_ptr + i * K + 0, (q0 - s0 * s0 / nn) * tl.load(inv_ptr + 0) * scale, mask=live)
    tl.store(cost_ptr + i * K + 1, (q1 - s1 * s1 / nn) * tl.load(inv_ptr + 1) * scale, mask=live)
    tl.store(cost_ptr + i * K + 2, (q2 - s2 * s2 / nn) * tl.load(inv_ptr + 2) * scale, mask=live)
    tl.store(on_ptr + i, last_in.to(tl.int8), mask=live)


def geo_cost(pos, te, tn, y, maps, mask, inv_var, ext, res, min_inside=8, block=128):
    """pos (N,2) particle E/N in metres (float64 is fine), te/tn (L,) track offsets, y (3,L) measured
    profiles, maps (3,H,W), mask (H,W) uint8, inv_var (3,), all on the GPU.
    Returns cost (N,3), on_map (N,) bool.

    UTM northings (~3.9e6 m) only resolve to ~0.25 m in float32, so positions are converted to
    local map coordinates (metres from the north-west corner) before going to float32."""
    N, L = pos.shape[0], te.shape[0]
    K, H, W = maps.shape
    assert K == 3
    cost = torch.empty((N, K), device=pos.device, dtype=torch.float32)
    on = torch.empty(N, device=pos.device, dtype=torch.int8)
    px = (pos[:, 0] - ext.x0).float().contiguous()
    py = (ext.y1 - pos[:, 1]).float().contiguous()
    _geo_cost_kernel[(triton.cdiv(N, block),)](
        px, py, te, tn, y, maps, mask, inv_var, cost, on, N, H, W,
        1.0 / res, float(min_inside), L=L, K=K, BLOCK=block)
    return cost, on.bool()
