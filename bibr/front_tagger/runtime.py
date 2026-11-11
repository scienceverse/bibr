"""ONNX Runtime front-matter tagger: encoder emissions + numpy CRF.

The bundle is an ``onnx/`` directory like the reference parser's: the manifest
(``bibr_onnx.json``) carries the tag set, the feature keys the model was
trained on and the CRF parameters; ``model.onnx`` maps ``input_ids``,
``attention_mask`` and ``token_features`` to ``emissions``.

The input string is tokenized without special tokens. Each subtoken takes the
feature row of the first word its offsets overlap (a zero row between
regions), and a word's tag and confidence are those of the first subtoken that
starts in it. A word no subtoken starts in (``","`` folded into ``"Smith,"``,
most kanji under a SentencePiece vocabulary) takes the ``I-`` form of the tag
of the subtoken that covers it, the label training gives it. Inputs longer
than the bundle's ``max_length`` run in overlapping windows; every token keeps
the emissions of the window it sits deepest in, and one CRF pass decodes the
stitched sequence.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bibr.front_tagger.contract import CONTRACT_VERSION, FEATURE_KEYS, TaggerInput
from bibr.ner.crf_numpy import marginals, viterbi_decode
from bibr.utils.ml_runtime import ONNX_MODEL, read_onnx_manifest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bibr.config import GlobalSettings
    from bibr.extract.front_role_features import FrontRegion
    from bibr.front_tagger.assemble import TaggedFrontMatter

logger = logging.getLogger(__name__)


def _words_of_tokens(offsets: list[tuple[int, int]], inp: TaggerInput) -> list[list[int]]:
    """Indexes of the words each token overlaps, in order; empty for whitespace
    and region breaks."""
    out: list[list[int]] = []
    w = 0
    words = inp.words
    for start, end in offsets:
        while w < len(words) and words[w].end <= start:
            w += 1
        hit: list[int] = []
        k = w
        while k < len(words) and words[k].start < end:
            if words[k].end > start:
                hit.append(k)
            k += 1
        out.append(hit)
    return out


def _inside(tag: str) -> str:
    return tag if tag == "O" else f"I-{tag[2:]}"


def _word_tags(
    words_of: list[list[int]], token_tags: list[str], token_conf: list[float], n_words: int
) -> tuple[list[str], list[float]]:
    """Word tags from token tags: a word's own first subtoken decides; a word
    only covered by a token that starts in an earlier word continues it."""
    tags = ["O"] * n_words
    conf = [1.0] * n_words
    own: set[int] = set()
    covered: set[int] = set()
    for hit, tag, p in zip(words_of, token_tags, token_conf, strict=True):
        if not hit:
            continue
        first = hit[0]
        if first not in own:
            own.add(first)
            tags[first], conf[first] = tag, p
        for w in hit[1:]:
            if w not in own and w not in covered:
                covered.add(w)
                tags[w], conf[w] = _inside(tag), p
    return tags, conf


def _windows(length: int, size: int) -> list[tuple[int, int]]:
    if length <= size:
        return [(0, length)]
    stride = max(1, size * 3 // 4)
    spans = []
    start = 0
    while True:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            return spans
        start += stride


class OnnxFrontTagger:
    runtime = "onnx"

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        from bibr.exceptions import ConfigurationError
        from bibr.utils.onnx_providers import create_session
        from bibr.utils.onnx_tokenizer import load_tokenizer

        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_onnx_manifest(self.bundle_dir)
        keys = tuple(self.manifest.get("feature_keys") or ())
        if keys != FEATURE_KEYS or self.manifest.get("contract_version") != CONTRACT_VERSION:
            raise ConfigurationError(
                f"Front-matter tagger bundle {self.bundle_dir} was trained on another input "
                f"contract (version {self.manifest.get('contract_version')!r}); this bibr "
                f"builds version {CONTRACT_VERSION}"
            )
        self.tags: list[str] = [str(t) for t in self.manifest["bio_tags"]]
        crf = self.manifest["crf"]
        self.start = np.asarray(crf["start_transitions"], dtype=np.float32)
        self.end = np.asarray(crf["end_transitions"], dtype=np.float32)
        self.transitions = np.asarray(crf["transitions"], dtype=np.float32)
        if self.transitions.shape != (len(self.tags), len(self.tags)):
            raise ConfigurationError(
                f"Front-matter tagger bundle {self.bundle_dir}: transitions shape "
                f"{self.transitions.shape} does not match {len(self.tags)} tags"
            )
        self.max_length = int(self.manifest.get("max_length", 4096))
        self.tokenizer = load_tokenizer(self.bundle_dir, self.manifest)
        # Whole-input encoding: the windows below do the cutting.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.session, self.device = create_session(
            self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL),
            device=device,
            model_name="front-matter tagger (ONNX)",
        )
        self.version = str(self.manifest.get("model_version") or self.bundle_dir)

    def _emissions(self, ids: np.ndarray, feats: np.ndarray) -> np.ndarray:
        feeds = {
            "input_ids": ids[None, :].astype(np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "token_features": feats[None, :, :].astype(np.float32),
        }
        (emissions,) = self.session.run(["emissions"], feeds)
        return np.asarray(emissions, dtype=np.float32)[0]

    def _token_inputs(self, inp: TaggerInput) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
        """``(token ids, words of each token, token feature rows)`` of ``inp``."""
        encoding = self.tokenizer.encode(inp.text, add_special_tokens=False)
        ids = np.asarray(encoding.ids, dtype=np.int64)
        words_of = _words_of_tokens(list(encoding.offsets), inp)
        feats = np.zeros((len(ids), len(FEATURE_KEYS)), dtype=np.float32)
        rows = np.asarray(inp.features, dtype=np.float32)
        for t, hit in enumerate(words_of):
            if hit:
                feats[t] = rows[hit[0]]
        return ids, words_of, feats

    def check_graph(self) -> None:
        """Run the graph once on a full window of front matter, as ``tag`` sends it.

        The load probe of an overridden tagger (:mod:`bibr.utils.onnx_probe`):
        the regions of a paper's first two pages through
        :func:`~bibr.front_tagger.contract.build_tagger_input`, the bundle's
        tokenizer and the feature rows ``tag`` builds, repeated to the
        manifest's ``max_length`` (the longest window a paper sends), with the
        tokenizer's last id in it. The graph must give finite emissions, one per
        tag of the manifest for every token, and the CRF's scores must match the
        tag count; the window is decoded as a paper's is. Raises
        :class:`~bibr.exceptions.ConfigurationError`.
        """
        from bibr.exceptions import ConfigurationError
        from bibr.front_tagger.contract import build_tagger_input
        from bibr.ner.crf_numpy import marginals, viterbi_decode
        from bibr.utils.onnx_probe import check_crf, check_output, last_token_id, run_probe

        check_crf(self.bundle_dir, self.tags, self.start, self.end, self.transitions)
        ids, _words_of, feats = self._token_inputs(build_tagger_input(_probe_regions()))
        if not len(ids):
            raise ConfigurationError(
                f"{self.bundle_dir}: the tokenizer gives no tokens for a paper's front matter"
            )
        positions = np.arange(self.max_length) % len(ids)
        ids, feats = ids[positions], feats[positions]
        if (last := last_token_id(self.tokenizer)) >= 0:
            ids[-1] = last
        model_path = self.bundle_dir / self.manifest.get("model_file", ONNX_MODEL)
        shape = f"a (1, {len(ids)}) window of front matter with {len(FEATURE_KEYS)} features"
        emissions = check_output(
            model_path,
            "emissions",
            run_probe(model_path, shape, lambda: self._emissions(ids, feats)[None]),
            (1, len(ids), len(self.tags)),
            shape,
            f"one score per tag of the manifest's {len(self.tags)} bio_tags",
        )[0]
        crf = {
            "start_transitions": self.start,
            "end_transitions": self.end,
            "transitions": self.transitions,
        }
        viterbi_decode(emissions[None], np.ones((1, len(ids)), dtype=bool), **crf)
        marginals(emissions, **crf)

    def tag(self, inp: TaggerInput) -> tuple[list[str], list[float]]:
        """One BIO tag and its posterior per word of ``inp``."""
        if not inp.words:
            return [], []
        ids, words_of, feats = self._token_inputs(inp)

        emissions = np.zeros((len(ids), len(self.tags)), dtype=np.float32)
        depth = np.full(len(ids), -1, dtype=np.int64)
        for lo, hi in _windows(len(ids), self.max_length):
            window = self._emissions(ids[lo:hi], feats[lo:hi])
            for offset in range(hi - lo):
                t = lo + offset
                here = min(offset, hi - lo - 1 - offset)
                if here > depth[t]:
                    depth[t] = here
                    emissions[t] = window[offset]

        crf = {
            "start_transitions": self.start,
            "end_transitions": self.end,
            "transitions": self.transitions,
        }
        path = viterbi_decode(emissions[None], np.ones((1, len(ids)), dtype=bool), **crf)[0]
        posterior = marginals(emissions, **crf)

        return _word_tags(
            words_of,
            [self.tags[k] for k in path],
            [float(posterior[t, k]) for t, k in enumerate(path)],
            len(inp.words),
        )


def _probe_regions() -> list[FrontRegion]:
    """The layout regions of a paper's first two pages, for the load probe."""
    from bibr.extract.front_role_features import FrontRegion
    from bibr.utils.onnx_probe import PROBE_BODY

    rows = [
        (0, "header", "Journal of Examples 12 (2021) 45–67", 0.04, 8.0),
        (
            0,
            "doc_title",
            "Memory consolidation during sleep: a preregistered replication",
            0.1,
            16.0,
        ),
        (0, "text", "Jana Nováková¹*, Kenji Ōtsuka², and María García-López¹", 0.18, 10.0),
        (0, "text", "¹ Department of Psychology, Example University, Brno, Czechia", 0.23, 8.0),
        (0, "text", "* Corresponding author: jana.novakova@example.org", 0.27, 8.0),
        (0, "abstract", "Abstract. " + PROBE_BODY * 2, 0.32, 9.0),
        (0, "text", "Keywords: sleep; memory; replication", 0.6, 9.0),
        (0, "text", PROBE_BODY * 2, 0.66, 10.0),
        (0, "footer", "https://doi.org/10.1000/xyz123 © 2021 The Authors. CC BY 4.0", 0.95, 7.0),
        (1, "text", PROBE_BODY * 4, 0.08, 10.0),
    ]
    return [
        FrontRegion(
            page=page,
            index=index,
            label=label,
            text=text,
            x0=0.1,
            y0=top,
            x1=0.9,
            y1=min(top + 0.04 + len(text) / 6000, 0.99),
            font_size=size,
            font_bold=label == "doc_title",
        )
        for index, (page, label, text, top, size) in enumerate(rows)
    ]


_CACHE: dict[tuple[str, str | None, str | None], OnnxFrontTagger | None] = {}
_CACHE_LOCK = threading.Lock()
# One load at a time, taken before ``_CACHE_LOCK``: chunks that ask for the tagger
# together would each read the bundle (about half a gigabyte) and open a session.
_LOAD_LOCK = threading.Lock()


ENV_MODEL_ID = "ML_FRONT_TAGGER_MODEL_ID"


def load_front_tagger(
    settings: GlobalSettings, device: str | None = None
) -> OnnxFrontTagger | None:
    """The configured tagger, or ``None`` when it is off or cannot be loaded.

    ``settings`` are the run's (a fast run turns the tagger on for itself) and
    ``device`` is its ``--device``; macOS always uses the CPU provider, as CoreML
    is untested on this model. Like the region-role classifier, the tagger is
    optional: a missing bundle logs once and the run goes on without its answer,
    which a full run covers with the LLM and a fast run reports as a warning. A
    network failure is not remembered, so the next paper tries again. An
    overridden bundle (:mod:`bibr.utils.model_overrides`) that cannot be loaded,
    for any reason, raises :class:`~bibr.exceptions.ModelOverrideError` instead.

    Blocking: the first call reads the bundle (a download, on a first run) and
    opens the session, so async callers run it off the event loop.
    """
    from bibr.utils.ml_runtime import find_onnx_bundle

    ml = settings.ml
    model_id = getattr(ml, "front_tagger_mode", "off") != "off" and getattr(
        ml, "front_tagger_model_id", None
    )
    if not model_id:
        return None
    revision = getattr(ml, "front_tagger_revision", None) or None
    if sys.platform == "darwin":
        device = "cpu"
    key = (str(model_id), revision, str(device) if device else None)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _note_use(settings, _CACHE[key])
    from bibr.exceptions import ModelOverrideError
    from bibr.utils.model_overrides import is_model_override, override_load_error

    with _LOAD_LOCK:
        with _CACHE_LOCK:  # loaded by the call this one waited for
            if key in _CACHE:
                return _note_use(settings, _CACHE[key])
        tagger: OnnxFrontTagger | None = None
        try:
            bundle = find_onnx_bundle(
                str(model_id), revision, label="front-matter tagger", raise_transient=True
            )
            if bundle is None:
                missing = FileNotFoundError(f"no ONNX bundle for {model_id}")
                if (strict := override_load_error(settings, ENV_MODEL_ID, missing)) is not None:
                    raise strict
                logger.warning(
                    "Front-matter tagger %s has no ONNX bundle; the run goes on without it",
                    model_id,
                )
            else:
                tagger = OnnxFrontTagger(bundle, device)
                if is_model_override(settings, ENV_MODEL_ID):
                    # A candidate's graph runs once at load; the default is not probed.
                    tagger.check_graph()
                logger.info("Front-matter tagger loaded")
                logger.debug("Front-matter tagger bundle: %s", bundle)
        except ModelOverrideError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional model, never fatal
            # An overridden bundle that cannot be loaded stops the run, whatever the
            # reason; the default goes on without it (a network failure retried).
            if (strict := override_load_error(settings, ENV_MODEL_ID, exc)) is not None:
                raise strict from exc
            logger.warning(
                "Front-matter tagger %s unavailable (%s); the run goes on without it", model_id, exc
            )
            from bibr.utils.transient import is_transient_network_error

            if is_transient_network_error(exc):
                return None
        with _CACHE_LOCK:
            _CACHE[key] = tagger
    return _note_use(settings, tagger)


def _note_use(settings: GlobalSettings, tagger: OnnxFrontTagger | None) -> OnnxFrontTagger | None:
    """Write an overridden tagger into the run's record (:mod:`bibr.utils.model_overrides`)."""
    if tagger is not None:
        from bibr.utils.model_overrides import (
            is_model_override,
            onnx_bundle_files,
            record_model_load,
        )

        bundle = getattr(tagger, "bundle_dir", None)
        if bundle is not None and is_model_override(settings, ENV_MODEL_ID):
            record_model_load(settings, ENV_MODEL_ID, lambda: onnx_bundle_files(bundle))
    return tagger


# Later windows read for an abstract the first pages lack: a title-page
# manuscript or a preprint cover sheet prints it on page 3 or 4.
ABSTRACT_WINDOW_STARTS = (1, 2)


def tag_front_matter(
    tagger: OnnxFrontTagger,
    regions: Sequence[FrontRegion],
    *,
    first_page_index: int = 0,
    rules: bool = False,
) -> TaggedFrontMatter:
    """Tag the first pages; with no abstract there, take it from a later window.

    ``rules`` is passed to :func:`~bibr.front_tagger.assemble.assemble`: the
    readings for a run with no LLM behind the tagger.

    The tagger reads ``MAX_PAGES`` pages from ``first_page_index``. When that
    window holds no abstract, the windows starting one and then two pages
    later are tagged with the same model and the first abstract one yields
    replaces the missing one; every other field keeps the first window's
    answer, also when a later window fails to run.
    """
    from bibr.front_tagger.assemble import assemble
    from bibr.front_tagger.contract import build_tagger_input

    inp = build_tagger_input(regions, first_page_index=first_page_index)
    result = assemble(inp, *tagger.tag(inp), rules=rules)
    if "no_abstract" not in result.problems or result.title_keywords is None:
        return result
    pages = {region.page for region in regions}
    for offset in ABSTRACT_WINDOW_STARTS:
        start = first_page_index + offset
        if start not in pages:
            continue  # a blank page does not end the paper
        later_inp = build_tagger_input(regions, first_page_index=start)
        if not later_inp.words:
            continue
        try:
            later = assemble(later_inp, *tagger.tag(later_inp), rules=rules)
        except Exception:  # noqa: BLE001 - the first window's answer still stands
            logger.warning(
                "Front-matter tagger failed on the window from page %d; keeping the first window",
                start,
                exc_info=True,
            )
            continue
        abstract = later.title_keywords.abstract if later.title_keywords else None
        if not (abstract or "").strip():
            continue
        problems = [p for p in result.problems if p != "no_abstract"]
        problems += [p for p in later.problems if p == "short_abstract"]
        confidence = dict(result.confidence)
        if "abstract" in later.confidence:
            confidence["abstract"] = later.confidence["abstract"]
        return replace(
            result,
            title_keywords=result.title_keywords.model_copy(update={"abstract": abstract}),
            confidence=confidence,
            problems=problems,
        )
    return result


def release_front_tagger(tagger: OnnxFrontTagger) -> None:
    """Drop the cache's hold on ``tagger`` so its session can be freed.

    The cache is process-wide and otherwise keeps a loaded model (about half a
    gigabyte) for the life of the process. Only the cache's reference goes: a
    pipeline still holding the tagger keeps it, and the next load reads the
    bundle again.
    """
    with _CACHE_LOCK:
        for key in [key for key, value in _CACHE.items() if value is tagger]:
            del _CACHE[key]


def reset_front_tagger_cache() -> None:
    """Test hook: forget cached taggers (and cached failures)."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "ABSTRACT_WINDOW_STARTS",
    "OnnxFrontTagger",
    "load_front_tagger",
    "release_front_tagger",
    "reset_front_tagger_cache",
    "tag_front_matter",
]
