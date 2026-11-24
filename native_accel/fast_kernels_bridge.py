# -*- coding: utf-8 -*-
"""
native_accel/fast_kernels_bridge.py — Zero-PyTorch ctypes Bridge to `fast_brain3d_kernels.dll`
==============================================================================================
Barrier 1 Guarantee: Strictly Zero-PyTorch (`"torch" not in sys.modules`).
Automatically falls back cleanly if the compiled native library is unavailable.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np

_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_brain3d_kernels.dll")
_LIB: Optional[ctypes.CDLL] = None

if os.name == "nt" and os.path.isfile(_DLL_PATH):
    try:
        _lib = ctypes.CDLL(_DLL_PATH)
        _u8p = ctypes.POINTER(ctypes.c_uint8)
        _fp = ctypes.POINTER(ctypes.c_float)
        _ip = ctypes.POINTER(ctypes.c_int)

        _lib.ub3d_astar_tactical_step.argtypes = [
            _u8p, _fp, _fp, _fp,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            _fp, _fp, _fp, _ip,
        ]
        _lib.ub3d_astar_tactical_step.restype = ctypes.c_int

        _lib.ub3d_render_fovea_16x16.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            _fp, ctypes.c_int, ctypes.c_float,
            _fp, ctypes.c_int,
            _fp,
        ]
        _lib.ub3d_render_fovea_16x16.restype = None

        _lib.ub3d_directional_clearance_and_surface.argtypes = [
            _u8p, _fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, _fp, _fp,
        ]
        _lib.ub3d_directional_clearance_and_surface.restype = None

        _lib.ub3d_smooth_local_step.argtypes = [
            _u8p, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            _fp,
        ]
        _lib.ub3d_smooth_local_step.restype = None

        _lib.ub3d_integrate_first_person_frame.argtypes = [
            _u8p, _fp, _fp, _fp, _fp, _fp, _fp, _fp,
            ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            _fp, _fp, _fp, _fp,
        ]
        _lib.ub3d_integrate_first_person_frame.restype = None

        _lib.ub3d_select_frontier_cell.argtypes = [
            _u8p, _fp, _fp, _fp, _fp, _fp,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _ip, _ip, _ip,
        ]
        _lib.ub3d_select_frontier_cell.restype = ctypes.c_int
        _LIB = _lib
    except Exception:
        _LIB = None

NATIVE_KERNELS_AVAILABLE = _LIB is not None
_U8P = ctypes.POINTER(ctypes.c_uint8)
_FP = ctypes.POINTER(ctypes.c_float)


def u8_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(_U8P)


def f32_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(_FP)


def get_native_lib() -> Optional[ctypes.CDLL]:
    return _LIB
