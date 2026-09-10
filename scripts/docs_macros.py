"""mkdocs-macros hook: expose live code facts as {{ variables }} in pages."""


def define_env(env):
    from bibr.config import LlmOptions
    from bibr.export.json_export import _SCHEMA_VERSION
    from bibr.input.supported_files import SupportedFileType

    # Read class defaults directly rather than instantiating GlobalSettings():
    # instantiating reads the doc-builder's own environment / cwd `.env`, which
    # would bake a developer's local overrides (e.g. a personal LLM_PROVIDER)
    # into the published reference pages instead of the shipped defaults.
    env.variables["schema_version"] = _SCHEMA_VERSION
    env.variables["default_llm_provider"] = LlmOptions.model_fields["provider"].default
    env.variables["default_llm_model"] = LlmOptions.model_fields["model"].default
    # Backticked extension list in enum-declaration order, e.g.
    # "`.pdf`/`.docx`/`.xml`/`.html`/`.htm`/`.epub`". Rendered wherever the docs
    # enumerate the accepted file extensions, so the list can never drift from
    # SupportedFileType.
    env.variables["supported_extensions"] = "/".join(f"`{ft.value}`" for ft in SupportedFileType)
