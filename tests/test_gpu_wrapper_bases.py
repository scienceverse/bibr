"""The local and serve GPU wrappers must share their model-core implementation.

Local (``bibr.local.*``) and serve (``bibr.serve.deployments.*``) variants
differ only in lifecycle (unload vs batcher); the inference core —
postprocessing, decode+detect, OOM-halving splits — lives once in the shared
base classes. These identity checks pin the dedup so the pairs can't drift
apart again.
"""

import pytest

pytest.importorskip("numpy")


class TestLayoutDetectorBase:
    def _classes(self):
        from bibr.layout_base import BaseLayoutDetector
        from bibr.local.layout import LayoutDetector as LocalDetector
        from bibr.serve.deployments.layout import LayoutDetector as ServeDetector

        return BaseLayoutDetector, LocalDetector, ServeDetector

    def test_both_variants_subclass_the_base(self):
        base, local, serve = self._classes()
        assert issubclass(local, base)
        assert issubclass(serve, base)

    def test_inference_core_is_shared_not_copied(self):
        base, local, serve = self._classes()
        for method in ("_postprocess", "_detect_images", "_detect_pytorch"):
            assert getattr(local, method) is getattr(base, method)
            assert getattr(serve, method) is getattr(base, method)

    def test_lifecycle_stays_variant_specific(self):
        _, local, serve = self._classes()
        assert hasattr(local, "unload")
        assert not hasattr(serve, "unload")
        assert hasattr(serve, "aclose")


class TestSentenceSegmenterBase:
    def _classes(self):
        from bibr.local.segmenter import SentenceSegmenter as LocalSegmenter
        from bibr.segmenter_base import BaseSentenceSegmenter
        from bibr.serve.deployments.segmenter import SentenceSegmenter as ServeSegmenter

        return BaseSentenceSegmenter, LocalSegmenter, ServeSegmenter

    def test_both_variants_subclass_the_base(self):
        base, local, serve = self._classes()
        assert issubclass(local, base)
        assert issubclass(serve, base)

    def test_split_core_is_shared_not_copied(self):
        base, local, serve = self._classes()
        for method in ("_split_batch_with_retry", "_collect_split_result", "_split_many"):
            assert getattr(local, method) is getattr(base, method)
            assert getattr(serve, method) is getattr(base, method)

    def test_lifecycle_stays_variant_specific(self):
        _, local, serve = self._classes()
        assert hasattr(local, "unload")
        assert not hasattr(serve, "unload")
        assert hasattr(serve, "aclose")
