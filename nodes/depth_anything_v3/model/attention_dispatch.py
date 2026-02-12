"""Centralized attention dispatch with SageAttention backend support.

Detects available attention backends at import time and provides a unified
dispatch_attention() function that routes to the active backend.

Backends:
- sdpa: PyTorch's F.scaled_dot_product_attention (always available)
- sage2: SageAttention v2 (INT8 Q/K quantization, ~2x speedup)
- sage3: SageAttention v3 for Blackwell GPUs (FP4 quantization, ~3x speedup)
"""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("dinov2")

# Detect available backends at import time
_AVAILABLE_BACKENDS = {"sdpa"}

try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
    _AVAILABLE_BACKENDS.add("sage3")
except (ImportError, ModuleNotFoundError):
    pass

try:
    from sageattention import sageattn  # noqa: F401
    _AVAILABLE_BACKENDS.add("sage2")
except (ImportError, ModuleNotFoundError):
    pass

# Module-level active backend (set once at model load time)
_active_backend = "sdpa"


def set_backend(name: str) -> None:
    """Set the active attention backend.

    Falls back to sdpa with a warning if the requested backend is unavailable.
    """
    global _active_backend
    if name not in ("sdpa", "sage2", "sage3"):
        logger.warning(f"Unknown attention backend '{name}', falling back to sdpa")
        _active_backend = "sdpa"
        return

    if name not in _AVAILABLE_BACKENDS:
        logger.warning(
            f"Attention backend '{name}' requested but not installed, falling back to sdpa. "
            f"Available backends: {sorted(_AVAILABLE_BACKENDS)}"
        )
        _active_backend = "sdpa"
        return

    _active_backend = name
    logger.info(f"Attention backend set to: {_active_backend}")


def get_backend() -> str:
    """Return the name of the currently active attention backend."""
    return _active_backend


def get_available_backends():
    """Return set of available backend names."""
    return _AVAILABLE_BACKENDS.copy()


@torch.compiler.disable
def _sage3_attention(q, k, v, orig_dtype):
    """Run sage3 attention, hidden from torch.compile's dynamo tracer."""
    from sageattn3 import sageattn3_blackwell
    if q.dtype == torch.float32:
        q, k, v = q.half(), k.half(), v.half()
    out = sageattn3_blackwell(q, k, v, is_causal=False)
    return out.to(orig_dtype) if out.dtype != orig_dtype else out


@torch.compiler.disable
def _sage2_attention(q, k, v, orig_dtype):
    """Run sage2 attention, hidden from torch.compile's dynamo tracer."""
    from sageattention import sageattn
    if q.dtype == torch.float32:
        q, k, v = q.half(), k.half(), v.half()
    out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    return out.to(orig_dtype) if out.dtype != orig_dtype else out


def dispatch_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """Route attention computation to the active backend.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        attn_mask: Optional attention mask. When provided, always uses SDPA
                   since sage backends don't reliably handle masks.
        dropout_p: Dropout probability.

    Returns:
        Attention output tensor with same shape as q.
    """
    # Sage backends don't support attention masks — fall back to SDPA
    if attn_mask is not None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

    if _active_backend == "sage3":
        return _sage3_attention(q, k, v, q.dtype)

    if _active_backend == "sage2":
        return _sage2_attention(q, k, v, q.dtype)

    # Default: sdpa
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
