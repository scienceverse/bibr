"""Per-instance settings injection for LLMClient.

Two clients constructed with different ``GlobalSettings`` instances must read
their own values; a client constructed without one takes an isolated snapshot
of the process defaults.
"""

from types import SimpleNamespace

from bibr.clients.llm import LLMClient
from bibr.config import GlobalSettings, Settings


def _settings(**llm_overrides) -> GlobalSettings:
    s = GlobalSettings()
    for key, value in llm_overrides.items():
        setattr(s.llm, key, value)
    return s


class TestLLMClientSettingsInjection:
    def test_two_clients_read_their_own_settings(self):
        s1 = _settings(provider="openai", model="model-one")
        s2 = _settings(provider="anthropic", model="model-two")

        c1 = LLMClient(settings=s1)
        c2 = LLMClient(settings=s2)

        assert c1._settings_signature() == (
            "openai",
            s1.llm.base_url,
            s1.llm.api_key,
            "model-one",
        )
        assert c2._settings_signature() == (
            "anthropic",
            s2.llm.base_url,
            s2.llm.api_key,
            "model-two",
        )

    def test_usage_recorded_under_instance_model(self):
        c1 = LLMClient(settings=_settings(model="model-one"))
        c2 = LLMClient(settings=_settings(model="model-two"))

        completion = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        )
        c1._record_usage(completion)
        c2._record_usage(completion)

        assert list(c1.usage) == ["model-one"]
        assert list(c2.usage) == ["model-two"]
        assert c1.usage["model-one"]["total_tokens"] == 15

    def test_track_usage_from_instance_settings(self):
        on = _settings(track_usage=True)
        off = _settings(track_usage=False)

        assert LLMClient(settings=on)._track_usage is True
        assert LLMClient(settings=off)._track_usage is False

    def test_default_client_uses_isolated_snapshot(self):
        client = LLMClient()

        assert client._settings is not Settings
        assert client._settings_signature() == (
            Settings.llm.provider,
            Settings.llm.base_url,
            Settings.llm.api_key,
            Settings.llm.model,
        )
        assert client._track_usage is Settings.llm.track_usage

    def test_cap_input_uses_given_settings(self):
        small = _settings(max_input_chars=10)

        assert LLMClient._cap_input("x" * 50, small) == "x" * 10
        big = "x" * 50
        assert LLMClient._cap_input(big) == big

    def test_client_breaker_from_instance_settings(self):
        s = GlobalSettings()
        s.cb.failure_threshold = 99
        client = LLMClient(settings=s)

        assert client._breaker.failure_threshold == 99

    def test_get_client_selects_provider_from_instance_settings(self, monkeypatch):
        from bibr.clients import providers

        custom = _settings(provider="instance-provider")
        captured: list[tuple[str, GlobalSettings | None]] = []

        class InstanceProvider:
            name = "instance-provider"

            def __init__(self, settings=None):
                self._settings = settings

            def build_client(self):
                captured.append((self.name, self._settings))
                return object()

            def call_kwargs(self, reasoning_effort, max_tokens=None):
                return {}

        class GlobalProvider(InstanceProvider):
            name = "global-provider"

        monkeypatch.setitem(providers._PROVIDERS, InstanceProvider.name, InstanceProvider)
        monkeypatch.setitem(providers._PROVIDERS, GlobalProvider.name, GlobalProvider)
        monkeypatch.setattr(Settings.llm, "provider", GlobalProvider.name)

        LLMClient(settings=custom)._get_client()

        assert captured == [(InstanceProvider.name, custom)]

    def test_openai_call_kwargs_use_instance_settings(self):
        from bibr.clients import providers

        custom = _settings(provider="openai", base_url=None, max_tokens=1234)
        provider = providers.get("openai", settings=custom)

        assert provider.call_kwargs(reasoning_effort=None)["max_completion_tokens"] == 1234
