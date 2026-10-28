"""Project-owned AVX2 quantized matrix-vector and batched kernels.

The small native library is compiled locally with the installed MSVC toolchain;
it is not a third-party inference runtime. If compilation or AVX2 support is
unavailable, callers can fall back to the portable Numba implementation.
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
from pathlib import Path
import subprocess

import numpy as np

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / 'native_quant.cpp'
_DLL = _ROOT / 'native_quant_v5.dll'
_LIB = None
_TRIED = False
_LOG = logging.getLogger(__name__)


def _cpu_has_avx2() -> bool:
    try:
        features = np._core._multiarray_umath.__cpu_features__
        return bool(features.get('AVX2', False) and features.get('FMA3', False))
    except Exception:
        return False


def _build() -> bool:
    candidates = []
    for base in (os.environ.get('ProgramFiles', r'C:\Program Files'),
                 os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')):
        candidates.extend(glob.glob(str(Path(base) / 'Microsoft Visual Studio' / '*' / '*' / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat')))
    candidates.extend(glob.glob(r'C:\Program Files\Microsoft Visual Studio\*\Community\VC\Auxiliary\Build\vcvars64.bat'))
    if not candidates:
        return False
    vcvars = candidates[-1]
    obj = _ROOT / 'native_quant.obj'
    imp = _ROOT / 'native_quant.lib'
    command = (
        f'call "{vcvars}" >nul && '
        f'cl /nologo /O2 /fp:fast /arch:AVX2 /std:c++17 /EHsc /MT /LD '
        f'/Fo:"{obj}" /Fe:"{_DLL}" "{_SOURCE}" /link /IMPLIB:"{imp}"'
    )
    try:
        # shell=True delegates Windows quoting to cmd.exe. Passing the batch
        # call through a list with /s /c escapes the embedded quotes on paths
        # such as "C:\Program Files\..." and makes an otherwise valid build fail.
        result = subprocess.run(command, shell=True, cwd=str(_ROOT.parent),
                                capture_output=True, text=True, timeout=120,
                                check=False)
        if result.returncode != 0 or not _DLL.exists():
            _LOG.warning('Native AVX2 build failed (exit=%s): %s', result.returncode,
                         (result.stderr or result.stdout).strip()[-2000:])
            return False
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def get_native():
    global _LIB, _TRIED
    if _LIB is not None:
        return _LIB
    if _TRIED or not _cpu_has_avx2():
        return None
    _TRIED = True
    if (not _DLL.exists() or _SOURCE.stat().st_mtime > _DLL.stat().st_mtime) and not _build():
        return None
    try:
        lib = ctypes.CDLL(str(_DLL))
        byte_ptr = ctypes.POINTER(ctypes.c_uint8)
        float_ptr = ctypes.POINTER(ctypes.c_float)
        signature = [byte_ptr, float_ptr, float_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.localai_q8_0_matvec.argtypes = signature
        lib.localai_q8_0_matvec.restype = None
        lib.localai_q4_0_matvec.argtypes = signature
        lib.localai_q4_0_matvec.restype = None
        matmul_signature = [byte_ptr, float_ptr, float_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        for symbol in ('localai_q8_0_matmul', 'localai_q4_0_matmul', 'localai_q4k_matmul', 'localai_q5k_matmul', 'localai_q6k_matmul'):
            fn = getattr(lib, symbol)
            fn.argtypes = matmul_signature
            fn.restype = None
        lib.localai_q4k_matvec.argtypes = signature
        lib.localai_q4k_matvec.restype = None
        lib.localai_q5k_matvec.argtypes = signature
        lib.localai_q5k_matvec.restype = None
        lib.localai_q6k_matvec.argtypes = signature
        lib.localai_q6k_matvec.restype = None
        _LIB = lib
        _LOG.info('Inference kernel: project-owned native AVX2/FMA Q4_0 + Q8_0 + Q4_K + Q5_K + Q6_K (MSVC DLL)')
        return _LIB
    except (OSError, AttributeError):
        return None


def matvec(raw, x: np.ndarray, rows: int, cols: int, ggml_type: int, threads: int = 4):
    """Return a direct quantized matvec result, or None if native isn't usable."""
    if ggml_type not in (2, 8, 12, 13, 14) or rows < 1:
        return None
    block_size = 32 if ggml_type in (2, 8) else 256
    if cols < block_size or cols % block_size:
        return None
    lib = get_native()
    if lib is None:
        return None
    raw_view = np.frombuffer(raw, dtype=np.uint8)
    bytes_per_block = 18 if ggml_type == 2 else (34 if ggml_type == 8 else (144 if ggml_type == 12 else (176 if ggml_type == 13 else 210)))
    expected_bytes = rows * (cols // block_size) * bytes_per_block
    if raw_view.size != expected_bytes or np.asarray(x).size != cols:
        raise ValueError('native quantized matvec received an invalid tensor or vector size')
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)
    raw_ptr = raw_view.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    fn = (lib.localai_q4_0_matvec if ggml_type == 2 else lib.localai_q8_0_matvec if ggml_type == 8 else
          lib.localai_q4k_matvec if ggml_type == 12 else
          lib.localai_q5k_matvec if ggml_type == 13 else lib.localai_q6k_matvec)
    fn(raw_ptr, x_ptr, out_ptr, int(rows), int(cols), int(threads))
    return out



def matmul(raw, vectors: np.ndarray, rows: int, cols: int, ggml_type: int, threads: int = 4):
    """Batched packed-quant matmul; each quantized row stays hot across tokens."""
    if ggml_type not in (2, 8, 12, 13, 14) or rows < 1:
        return None
    block_size = 32 if ggml_type in (2, 8) else 256
    if cols < block_size or cols % block_size:
        return None
    lib = get_native()
    if lib is None:
        return None
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != cols or x.shape[0] < 1:
        raise ValueError('native quantized matmul received an invalid input matrix')
    raw_view = np.frombuffer(raw, dtype=np.uint8)
    bytes_per_block = 18 if ggml_type == 2 else (34 if ggml_type == 8 else (144 if ggml_type == 12 else (176 if ggml_type == 13 else 210)))
    expected_bytes = rows * (cols // block_size) * bytes_per_block
    if raw_view.size != expected_bytes:
        raise ValueError('native quantized matmul received an invalid tensor size')
    # Native output is [rows, tokens] for contiguous row writes; transpose once
    # at the boundary to the engine's [tokens, rows] convention.
    out = np.empty((rows, x.shape[0]), dtype=np.float32)
    raw_ptr = raw_view.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    symbol = {2: 'localai_q4_0_matmul', 8: 'localai_q8_0_matmul', 12: 'localai_q4k_matmul', 13: 'localai_q5k_matmul', 14: 'localai_q6k_matmul'}[ggml_type]
    getattr(lib, symbol)(raw_ptr, x_ptr, out_ptr, int(rows), int(cols), int(x.shape[0]), int(threads))
    return out.T.copy()

