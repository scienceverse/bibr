"""Utility modules — explicit-import only.

Submodules (``circuit_breaker``, ``rate_limiter``, ``semaphore``,
``device``, ``onnx_providers``, ``text``) have non-trivial import cost
or pull in optional deps. Import them directly rather than relying on
re-exports here.
"""
