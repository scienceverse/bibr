"""Sentence-segmenter backend factories."""

from bibr.segmenter import registry
from bibr.segmenter.registry import create, known_backends, register

__all__ = ["create", "known_backends", "register", "registry"]
