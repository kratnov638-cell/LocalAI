"""Optional JIT-compiled GGML block decoders.

These are small, project-owned kernels rather than an inference runtime. Numba
compiles the independent 256-value quant blocks and parallelizes them across
CPU cores; the NumPy implementation remains the correctness fallback.
"""
from numba import njit, prange
import numpy as np


@njit(cache=True, nogil=True)
def delta_net_step(state, q, k, v, decay, beta):
    """Fused Gated DeltaNet recurrent step for one token.

    The reference NumPy path separately scales the state, contracts it with k,
    applies a rank-one update, and contracts it with q. This kernel performs
    the same operations in two state passes and avoids multi-megabyte temporary
    arrays for the outer product on every generated token.
    """
    heads, key_dim, value_dim = state.shape
    core = np.zeros((heads, value_dim), dtype=np.float32)
    predicted = np.empty(value_dim, dtype=np.float32)
    delta = np.empty(value_dim, dtype=np.float32)
    for h in range(heads):
        factor = np.float32(np.exp(decay[h]))
        for vi in range(value_dim):
            predicted[vi] = 0.0
        # Pass 1: decay state in-place while calculating state @ k.
        for ki in range(key_dim):
            kval = k[h, ki]
            for vi in range(value_dim):
                value = state[h, ki, vi] * factor
                state[h, ki, vi] = value
                predicted[vi] += value * kval
        for vi in range(value_dim):
            delta[vi] = (v[h, vi] - predicted[vi]) * beta[h]
        # Pass 2: apply the rank-one update and calculate updated_state @ q.
        for ki in range(key_dim):
            kval = k[h, ki]
            qval = q[h, ki]
            for vi in range(value_dim):
                value = state[h, ki, vi] + kval * delta[vi]
                state[h, ki, vi] = value
                core[h, vi] += value * qval
    return core


@njit(cache=True, parallel=True)
def dequant_q4_k(packed, d, dmin):
    """Decode Q4_K superblocks; packed rows are the 144-byte GGML blocks."""
    blocks = packed.shape[0]
    out = np.empty((blocks, 256), dtype=np.float32)
    for b in prange(blocks):
        base_scale = 4
        base_qs = 16
        db = d[b]
        mb = dmin[b]
        for group in range(8):
            if group < 4:
                sc = int(packed[b, base_scale + group]) & 0x3F
                mn = int(packed[b, base_scale + group + 4]) & 0x3F
            else:
                sc = ((int(packed[b, base_scale + group + 4]) & 0x0F)
                      | ((int(packed[b, base_scale + group - 4]) >> 6) << 4))
                mn = ((int(packed[b, base_scale + group + 4]) >> 4)
                      | ((int(packed[b, base_scale + group]) >> 6) << 4))
            qs_base = base_qs + (group // 2) * 32
            shift = 4 if group % 2 else 0
            for j in range(32):
                q = (int(packed[b, qs_base + j]) >> shift) & 0x0F
                out[b, group * 32 + j] = db * sc * q - mb * mn
    return out


@njit(cache=True, parallel=True)
def dequant_q6_k(packed, d):
    """Decode Q6_K superblocks using the GGML ql/qh/scales layout."""
    blocks = packed.shape[0]
    out = np.empty((blocks, 256), dtype=np.float32)
    for b in prange(blocks):
        db = d[b]
        for half in range(2):
            ql_base = half * 64
            qh_base = 128 + half * 32
            scale_base = 192 + half * 8
            out_base = half * 128
            for quad in range(4):
                for j in range(32):
                    if quad == 0:
                        qlow = int(packed[b, ql_base + j]) & 0x0F
                        qhigh = int(packed[b, qh_base + j]) & 0x03
                    elif quad == 1:
                        qlow = int(packed[b, ql_base + 32 + j]) & 0x0F
                        qhigh = (int(packed[b, qh_base + j]) >> 2) & 0x03
                    elif quad == 2:
                        qlow = int(packed[b, ql_base + j]) >> 4
                        qhigh = (int(packed[b, qh_base + j]) >> 4) & 0x03
                    else:
                        qlow = int(packed[b, ql_base + 32 + j]) >> 4
                        qhigh = (int(packed[b, qh_base + j]) >> 6) & 0x03
                    q = int(qlow) | (int(qhigh) << 4)
                    signed_scale = int(packed[b, scale_base + quad * 2 + j // 16])
                    if signed_scale >= 128:
                        signed_scale -= 256
                    out[b, out_base + quad * 32 + j] = (q - 32) * db * signed_scale
    return out

@njit(cache=True, parallel=True, fastmath=True)
def matvec_q4_k_direct(packed, d, dmin, x):
    """Direct Q4_K matrix-vector product without expanding weights to float32."""
    rows, blocks_per_row, _ = packed.shape
    out = np.zeros(rows, dtype=np.float32)
    for row in prange(rows):
        total = 0.0
        for b in range(blocks_per_row):
            db = d[row, b]
            mb = dmin[row, b]
            block = packed[row, b]
            for group in range(8):
                if group < 4:
                    sc = int(block[4 + group]) & 0x3F
                    mn = int(block[8 + group]) & 0x3F
                else:
                    sc = (int(block[4 + group + 4]) & 0x0F) | ((int(block[4 + group - 4]) >> 6) << 4)
                    mn = (int(block[4 + group + 4]) >> 4) | ((int(block[4 + group]) >> 6) << 4)
                qs_base = 16 + (group // 2) * 32
                shift = 4 if group % 2 else 0
                x_base = b * 256 + group * 32
                scale = db * sc
                minimum = mb * mn
                for j in range(32):
                    q = (int(block[qs_base + j]) >> shift) & 0x0F
                    total += (scale * q - minimum) * x[x_base + j]
        out[row] = total
    return out


@njit(cache=True, parallel=True, fastmath=True)
def matvec_q6_k_direct(packed, d, x):
    """Direct Q6_K matrix-vector product without expanding weights to float32."""
    rows, blocks_per_row, _ = packed.shape
    out = np.zeros(rows, dtype=np.float32)
    for row in prange(rows):
        total = 0.0
        for b in range(blocks_per_row):
            db = d[row, b]
            block = packed[row, b]
            x_base_block = b * 256
            for half in range(2):
                ql_base = half * 64
                qh_base = 128 + half * 32
                scale_base = 192 + half * 8
                out_base = x_base_block + half * 128
                for quad in range(4):
                    for j in range(32):
                        if quad == 0:
                            qlow = int(block[ql_base + j]) & 0x0F
                            qhigh = int(block[qh_base + j]) & 0x03
                        elif quad == 1:
                            qlow = int(block[ql_base + 32 + j]) & 0x0F
                            qhigh = (int(block[qh_base + j]) >> 2) & 0x03
                        elif quad == 2:
                            qlow = int(block[ql_base + j]) >> 4
                            qhigh = (int(block[qh_base + j]) >> 4) & 0x03
                        else:
                            qlow = int(block[ql_base + 32 + j]) >> 4
                            qhigh = (int(block[qh_base + j]) >> 6) & 0x03
                        q = (int(qlow) | (int(qhigh) << 4)) - 32
                        scale = int(block[scale_base + quad * 2 + j // 16])
                        if scale >= 128:
                            scale -= 256
                        total += (q * db * scale) * x[out_base + quad * 32 + j]
        out[row] = total
    return out
