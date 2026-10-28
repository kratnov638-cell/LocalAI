"""Vectorized GGML block dequantizers implemented with NumPy only."""
import numpy as np

# GGML type IDs: (values per block, encoded bytes per block). F32/F16 are
# essential too: quantized GGUFs commonly keep norm weights and biases unquantized.
LAYOUT = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
}


def block_layout(t):
    if t not in LAYOUT:
        raise NotImplementedError(
            f'unsupported tensor type {t}; supported: F32(0), F16(1), Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7), Q8_0(8), Q4_K(12), Q5_K(13), Q6_K(14)'
        )
    return LAYOUT[t]


def _f16(raw):
    # Return a zero-copy view. Casting the *entire* raw block to float32 just
    # to read its scale used to convert every packed quant byte as if it were
    # half precision (up to 105 useless conversions per Q6_K block).
    return np.frombuffer(raw, dtype='<f2')


def dequantize(raw, t, shape):
    """Decode packed GGML blocks to a float32 NumPy tensor.

    Work is vectorized across every block. In particular, no Python loop runs
    once per 32/256 values; that would make multi-billion-parameter models
    impractical to load on a CPU.
    """
    block, bytes_per_block = block_layout(t)
    n = int(np.prod(shape, dtype=np.int64))
    if n % block:
        raise ValueError(f'{n} elements is not divisible by block size {block}')
    count = n // block
    if t == 0:
        return np.frombuffer(raw, dtype='<f4', count=n).copy().reshape(shape)
    if t == 1:
        return np.frombuffer(raw, dtype='<f2', count=n).astype(np.float32).reshape(shape)
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(count, bytes_per_block)

    # Q4_K and Q6_K dominate this checkpoint. Their custom JIT decoders process
    # independent 256-value blocks in parallel; keep the portable NumPy path
    # below as a fallback when Numba is not installed.
    if t in (12, 14):
        try:
            from .numba_kernels import dequant_q4_k, dequant_q6_k
        except ImportError:
            pass
        else:
            if t == 12:
                params = _f16(raw).reshape(count, 72)
                values = dequant_q4_k(packed, params[:, 0].astype(np.float32), params[:, 1].astype(np.float32))
            else:
                params = _f16(raw).reshape(count, 105)
                values = dequant_q6_k(packed, params[:, 104].astype(np.float32))
            return values.reshape(shape)

    if t == 2:  # Q4_0: fp16 d, 16 bytes of low/high nibbles, signed range [-8, 7]
        d = _f16(raw).reshape(count, 9)[:, 0].astype(np.float32)
        q = packed[:, 2:].astype(np.int16)
        values = np.concatenate((q & 0x0F, q >> 4), axis=1).astype(np.float32) - 8.0
        values *= d[:, None]

    elif t == 3:  # Q4_1: fp16 scale, fp16 minimum, then 16 packed nibble bytes
        params = _f16(raw).reshape(count, 10)
        d, m = params[:, 0].astype(np.float32), params[:, 1].astype(np.float32)
        q = packed[:, 4:].astype(np.int16)
        values = np.concatenate((q & 0x0F, q >> 4), axis=1).astype(np.float32)
        values = values * d[:, None] + m[:, None]

    elif t == 8:  # Q8_0: fp16 d followed by 32 signed int8 values
        d = _f16(raw).reshape(count, 17)[:, 0].astype(np.float32)
        q = np.frombuffer(raw, dtype=np.int8).reshape(count, bytes_per_block)[:, 2:]
        values = q.astype(np.float32) * d[:, None]

    elif t in (6, 7):  # Q5_0 / Q5_1: high-bit mask plus packed low nibbles
        qh_offset = 2 if t == 6 else 4
        qs_offset = qh_offset + 4
        params = _f16(raw).reshape(count, bytes_per_block // 2)
        d = params[:, 0].astype(np.float32)
        m = params[:, 1].astype(np.float32) if t == 7 else None
        qh = packed[:, qh_offset:qh_offset + 4]
        qs = packed[:, qs_offset:qs_offset + 16]
        low = np.concatenate((qs & 0x0F, qs >> 4), axis=1).astype(np.int16)
        shifts = np.arange(32, dtype=np.uint8)
        high = ((qh[:, shifts // 8] >> (shifts % 8)) & 1).astype(np.int16)
        q = low | (high << 4)
        if t == 6:
            values = (q.astype(np.float32) - 16.0) * d[:, None]
        else:
            values = q.astype(np.float32) * d[:, None] + m[:, None]

    elif t == 12:  # Q4_K: d, dmin, packed scale/min table, 128 quant bytes
        params = _f16(raw).reshape(count, 72)
        d = params[:, 0].astype(np.float32)
        dmin = params[:, 1].astype(np.float32)
        scales = packed[:, 4:16]
        qs = packed[:, 16:].reshape(count, 128)
        values = np.empty((count, 256), dtype=np.float32)
        # Each 32-byte qs chunk encodes 32 low-nibble values followed by
        # 32 high-nibble values in the next 32-value output group. Therefore
        # low/high groups are interleaved across the 256-value super-block.
        for group in range(8):
            if group < 4:
                sc = scales[:, group] & 0x3F
                mn = scales[:, group + 4] & 0x3F
            else:
                sc = (scales[:, group + 4] & 0x0F) | ((scales[:, group - 4] >> 6) << 4)
                mn = (scales[:, group + 4] >> 4) | ((scales[:, group] >> 6) << 4)
            qbytes = qs[:, (group // 2) * 32:(group // 2 + 1) * 32]
            q = (qbytes & 0x0F) if group % 2 == 0 else (qbytes >> 4)
            start = group * 32
            values[:, start:start + 32] = (
                q.astype(np.float32) * (d * sc.astype(np.float32))[:, None]
                - (dmin * mn.astype(np.float32))[:, None]
            )

    elif t == 13:  # Q5_K: d, dmin, packed scales/mins, high bits, low nibbles
        d = _f16(raw).reshape(count, 88)[:, 0]
        dmin = _f16(raw).reshape(count, 88)[:, 1]
        scales = packed[:, 4:16]
        qh = packed[:, 16:48]
        qs = packed[:, 48:176]
        values = np.empty((count, 256), dtype=np.float32)
        for group in range(8):
            if group < 4:
                sc = scales[:, group] & 0x3F
                mn = scales[:, group + 4] & 0x3F
            else:
                sc = (scales[:, group + 4] & 0x0F) | ((scales[:, group - 4] >> 6) << 4)
                mn = (scales[:, group + 4] >> 4) | ((scales[:, group] >> 6) << 4)
            qbytes = qs[:, (group // 2) * 32:(group // 2 + 1) * 32]
            qlow = (qbytes & 0x0F) if group % 2 == 0 else (qbytes >> 4)
            qhigh = (qh >> group) & 1
            q = qlow.astype(np.int16) | (qhigh.astype(np.int16) << 4)
            start = group * 32
            values[:, start:start + 32] = (
                q.astype(np.float32) * (d * sc.astype(np.float32))[:, None]
                - (dmin * mn.astype(np.float32))[:, None]
            )

    else:  # Q6_K: ql[128], qh[64], int8 scales[16], fp16 d
        ql = packed[:, :128].reshape(count, 2, 64)
        qh = packed[:, 128:192].reshape(count, 2, 32)
        scales = packed[:, 192:208].view(np.int8).reshape(count, 16)
        d = _f16(raw).reshape(count, 105)[:, 104].astype(np.float32)
        values = np.empty((count, 256), dtype=np.float32)
        for half in range(2):
            lo = ql[:, half, :]
            hi = qh[:, half, :]
            base = half * 128
            si = half * 8
            q1 = (lo[:, :32] & 0x0F) | ((hi & 0x03) << 4)
            q2 = (lo[:, 32:64] & 0x0F) | (((hi >> 2) & 0x03) << 4)
            q3 = (lo[:, :32] >> 4) | (((hi >> 4) & 0x03) << 4)
            q4 = (lo[:, 32:64] >> 4) | (((hi >> 6) & 0x03) << 4)
            quads = (q1, q2, q3, q4)
            for quad, q in enumerate(quads):
                scale = scales[:, si + quad * 2:si + quad * 2 + 2].repeat(16, axis=1)
                start = base + quad * 32
                values[:, start:start + 32] = (q.astype(np.int16) - 32).astype(np.float32) * d[:, None] * scale.astype(np.float32)

    return values.reshape(shape)
