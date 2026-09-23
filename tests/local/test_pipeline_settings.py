"""Per-instance settings injection for LocalPipeline.

Two pipelines with different ``GlobalSettings`` instances must coexist in one
process; the default (no settings arg) path must stay identical to before.
"""

from bibr.config import GlobalSettings, Settings, snapshot_settings
from bibr.local.pipeline import LocalPipeline


def test_snapshot_settings_is_deep_and_stable():
    source = GlobalSettings()
    source.crossref.enrich = True

    snapshot = snapshot_settings(source)
    source.crossref.enrich = False

    assert snapshot is not source
    assert snapshot.crossref is not source.crossref
    assert snapshot.crossref.enrich is True


def test_snapshot_settings_materializes_global_proxy(monkeypatch):
    monkeypatch.setattr(Settings.llm, "model", "snapshot-model")

    snapshot = snapshot_settings()
    monkeypatch.setattr(Settings.llm, "model", "mutated-model")

    assert isinstance(snapshot, GlobalSettings)
    assert snapshot.llm.model == "snapshot-model"


def test_pipeline_threads_settings_to_llm_client():
    custom = GlobalSettings()
    custom.llm.model = "custom-model-for-test"

    pipe = LocalPipeline(llm_backend="cloud", settings=custom)
    client = pipe._resources.llm_client

    assert client._settings is pipe.settings
    assert client._settings is not custom
    assert client._settings_signature()[3] == "custom-model-for-test"


def test_pipeline_resolves_defaults_from_instance_settings(monkeypatch):
    custom = GlobalSettings()
    custom.ocr.backend = "glm-http"
    custom.ocr.__pydantic_fields_set__.add("backend")
    custom.llm.backend = "llama-cpp"
    custom.pipeline.memory_mode = "keep_all"

    monkeypatch.setattr(Settings.ocr, "backend", "glm-llama")
    monkeypatch.setattr(Settings.llm, "backend", "cloud")
    monkeypatch.setattr(Settings.pipeline, "memory_mode", "aggressive")

    pipe = LocalPipeline(settings=custom)

    assert pipe.ocr_backend == "glm-http"
    assert pipe.llm_backend == "llama-cpp"
    assert pipe.memory_mode == "keep_all"
    assert pipe._resources._settings is pipe.settings
    assert pipe._resources._settings is not custom


def test_two_pipelines_coexist_with_different_settings():
    s1 = GlobalSettings()
    s1.llm.model = "model-one"
    s2 = GlobalSettings()
    s2.llm.model = "model-two"

    p1 = LocalPipeline(llm_backend="cloud", settings=s1)
    p2 = LocalPipeline(llm_backend="cloud", settings=s2)

    assert p1._resources.llm_client._settings.llm.model == "model-one"
    assert p2._resources.llm_client._settings.llm.model == "model-two"


def test_default_pipeline_keeps_lazy_global_client():
    pipe = LocalPipeline(llm_backend="cloud")

    assert pipe._resources._llm_client is None
    assert isinstance(pipe._resources.llm_client._settings, GlobalSettings)
    assert pipe._resources.llm_client._settings is pipe.settings


def test_pipeline_snapshot_is_stable_after_source_mutation():
    source = GlobalSettings()
    source.llm.model = "before"

    pipe = LocalPipeline(llm_backend="cloud", settings=source)
    source.llm.model = "after"

    assert pipe.settings.llm.model == "before"


def test_crossref_enrich_respects_instance_settings():
    def enrichers_of(pipe):
        # Cloud-LLM pipelines stream the back half: EnrichmentStage lives
        # inside the terminal composite stage rather than the top-level list.
        candidates = list(pipe._stages) + [s._enrich for s in pipe._stages if hasattr(s, "_enrich")]
        stage = next(s for s in candidates if type(s).__name__ == "EnrichmentStage")
        return stage._enrichers

    off = GlobalSettings()
    off.crossref.enrich = False
    off.ror.enrich = False
    on = GlobalSettings()
    on.crossref.enrich = True
    on.ror.enrich = False

    # ``crossref=None`` (the default) follows the instance's setting.
    assert enrichers_of(LocalPipeline(settings=off)) == []
    assert len(enrichers_of(LocalPipeline(settings=on))) == 1
    # An explicit per-pipeline switch wins over the setting either way.
    assert len(enrichers_of(LocalPipeline(crossref=True, settings=off))) == 1
    assert enrichers_of(LocalPipeline(crossref=False, settings=on)) == []


def test_crossref_enrich_is_off_by_default():
    """CROSSREF_ENRICH is opt-in: the field default is False, so a settings
    object built without the variable does not enrich."""
    from bibr.config import CrossrefOptions

    assert CrossrefOptions.model_fields["enrich"].default is False
    assert CrossrefOptions(_env_file=None, enrich=False).enrich is False


def test_global_settings_exported_from_bibr():
    import bibr

    assert bibr.GlobalSettings is GlobalSettings
