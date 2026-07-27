# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""One-time SDPA / flash-attention diagnostics.

Purely informational helpers that report which scaled-dot-product-attention
backends torch will use and whether the flash-attention kernel actually runs on
the current hardware/dtype. Nothing here changes training behavior.
"""

import os
import contextlib

import torch
import torch.nn.functional as F

# Guard so the flash-attention diagnostic below is only printed once per run.
_FLASH_ATTN_STATUS_PRINTED = False


def _get_sdpa_backend_flags() -> dict:
    """Report which scaled-dot-product-attention backends torch currently allows.

    Each probe is wrapped in try/except because the query API differs across
    torch versions; an unavailable flag is reported as None rather than crashing.
    """
    flags = {}
    try:
        flags["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        flags["flash"] = None
    try:
        flags["mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except Exception:
        flags["mem_efficient"] = None
    try:
        flags["math"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:
        flags["math"] = None
    return flags


def maybe_print_flash_attn_status_once(*, device: torch.device, dtype: torch.dtype, num_heads: int, head_dim: int, seqlen: int, rank: int) -> None:
    """One-time diagnostic: log torch/CUDA/GPU info and actively probe whether the
    flash-attention kernel can run with this model's attention shape.

    Runs only on rank 0, only once, and can be disabled via FLASH_ATTN_MONITOR=0.
    Purely informational — it never changes training behavior.
    """
    global _FLASH_ATTN_STATUS_PRINTED
    if _FLASH_ATTN_STATUS_PRINTED or rank != 0:
        return
    if os.environ.get("FLASH_ATTN_MONITOR", "1").strip().lower() in ("0", "false", "no", "off"):
        return

    _FLASH_ATTN_STATUS_PRINTED = True

    gpu_name = None
    try:
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
    except Exception:
        gpu_name = None

    flags = _get_sdpa_backend_flags()

    # Force a flash-attention-only context and run a tiny attention op to confirm
    # the kernel actually executes on this hardware/dtype (not just that it's enabled).
    probe_ok = False
    probe_err = None
    try:
        cm = contextlib.nullcontext()
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            cm = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        else:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                cm = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
            except Exception:
                cm = contextlib.nullcontext()

        with cm:
            q = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        probe_ok = True
    except Exception as e:
        probe_err = str(e).split("\n")[0]

    print(
        f"[SDPA] torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu_name} dtype={str(dtype).replace('torch.', '')} "
        f"flags={flags} flash_only_probe={{'ok': {probe_ok}, 'heads': {num_heads}, 'head_dim': {head_dim}, 'seqlen': {seqlen}, 'err': {probe_err!r}}}"
    )
