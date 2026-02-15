"""Centralized attention dispatch with configurable backends.

Provides a unified dispatch_attention() function that routes to the active backend.

Backends:
- sdpa: PyTorch's F.scaled_dot_product_attention (always available)
- flash_attn: Tri Dao's FlashAttention (FA2/FA3, requires flash-attn package)
- sage: SageAttention (auto-detects v3 for Blackwell or v2 for Ampere+)
"""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("DepthAnythingV3")

_active_backend = "sdpa"


def set_backend(name: str) -> str:
    """Set the active attention backend.

    Args:
        name: One of "sdpa", "flash_attn", "sage".
              For "sage", auto-detects sage3 (Blackwell) then sage2.

    Returns:
        The resolved backend name actually set (may differ if fallback occurred).
    """
    global _active_backend

    if name == "sdpa":
        _active_backend = "sdpa"
        logger.info("Attention backend: sdpa (PyTorch native)")
        return "sdpa"

    if name == "flash_attn":
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            _active_backend = "flash_attn"
            logger.info("Attention backend: flash_attn (Tri Dao's FlashAttention)")
            return "flash_attn"
        except ImportError:
            logger.warning("flash-attn package not installed, falling back to sdpa")
            _active_backend = "sdpa"
            return "sdpa"

    if name == "sage":
        # Try sage3 (Blackwell FP4) first, then sage2 (Ampere+ INT8)
        try:
            from sageattn3 import sageattn3  # noqa: F401
            _active_backend = "sage3"
            logger.info("Attention backend: sage3 (SageAttention v3, Blackwell FP4)")
            return "sage3"
        except ImportError:
            pass
        try:
            from sageattention import sageattn  # noqa: F401
            _active_backend = "sage2"
            logger.info("Attention backend: sage2 (SageAttention v2, INT8)")
            return "sage2"
        except ImportError:
            pass
        logger.warning("Neither sageattn3 nor sageattention installed, falling back to sdpa")
        _active_backend = "sdpa"
        return "sdpa"

    logger.warning(f"Unknown attention backend '{name}', falling back to sdpa")
    _active_backend = "sdpa"
    return "sdpa"


def get_backend() -> str:
    """Return the currently active attention backend name."""
    return _active_backend


def dispatch_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """Route attention to the active backend.

    Args:
        q: Query tensor (B, H, N, D)
        k: Key tensor (B, H, N, D)
        v: Value tensor (B, H, N, D)
        attn_mask: Optional attention mask. Forces SDPA when present.
        dropout_p: Dropout probability.

    Returns:
        Output tensor (B, H, N, D)
    """
    if attn_mask is not None or _active_backend == "sdpa":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

    if _active_backend == "flash_attn":
        return _dispatch_flash_attn(q, k, v, dropout_p)

    if _active_backend == "sage3":
        return _dispatch_sage3(q, k, v)

    if _active_backend == "sage2":
        return _dispatch_sage2(q, k, v)

    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def _dispatch_flash_attn(q, k, v, dropout_p):
    """FlashAttention expects (B, N, H, D); we receive (B, H, N, D)."""
    if q.dtype == torch.float32:
        logger.debug("flash_attn requires fp16/bf16, falling back to sdpa for this call")
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    try:
        from flash_attn import flash_attn_func
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q, k, v, dropout_p=dropout_p)
        return out.transpose(1, 2)
    except Exception as e:
        logger.warning(f"flash_attn failed ({e}), falling back to sdpa")
        return F.scaled_dot_product_attention(
            q.transpose(1, 2).transpose(1, 2), k.transpose(1, 2).transpose(1, 2),
            v.transpose(1, 2).transpose(1, 2), dropout_p=dropout_p
        )


@torch.compiler.disable
def _dispatch_sage3(q, k, v):
    """SageAttention v3 (Blackwell FP4). Hidden from torch.compile."""
    try:
        from sageattn3 import sageattn3
        return sageattn3(q, k, v)
    except Exception as e:
        logger.warning(f"sage3 failed ({e}), falling back to sdpa")
        return F.scaled_dot_product_attention(q, k, v)


@torch.compiler.disable
def _dispatch_sage2(q, k, v):
    """SageAttention v2 (INT8). Hidden from torch.compile."""
    try:
        from sageattention import sageattn
        return sageattn(q, k, v)
    except Exception as e:
        logger.warning(f"sage2 failed ({e}), falling back to sdpa")
        return F.scaled_dot_product_attention(q, k, v)
