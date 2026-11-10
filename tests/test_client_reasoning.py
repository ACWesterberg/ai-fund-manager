import sys
import types

import pytest

from fundmgr.config import AppConfig, LLMConfig
from fundmgr.engine import client as client_mod

_VALID_JSON = (
    '{"run_id":"t","market_summary":"m",'
    '"actions":[{"ticker":"AAA.ST","side":"buy","target_weight_pct":10,'
    '"sek_estimate":5000,"confidence":0.7,"thesis":"x"}],'
    '"cash_target_pct":8,"notes":""}'
)


class _Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


@pytest.fixture
def captured_anthropic(monkeypatch):
    """Inject a fake `anthropic` module; capture the create() kwargs."""
    captured = {}

    _final = types.SimpleNamespace(content=[
        # Mimic Opus-with-thinking: a thinking block precedes the text block.
        _Block("thinking", thinking="...reasoning..."),
        _Block("text", text=_VALID_JSON),
    ], stop_reason="end_turn", stop_details=None, model=None)
    captured["_final"] = _final

    class _StreamCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return _final

    class _Messages:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def stream(self, **kwargs):
            captured.update(kwargs, _endpoint=self.endpoint)
            if _final.model is None:
                _final.model = kwargs["model"]
            return _StreamCtx()

    class _Anthropic:
        def __init__(self, api_key=None, **_kw):
            self.messages = _Messages("stable")
            self.beta = types.SimpleNamespace(messages=_Messages("beta"))

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return captured


def _cfg(model_id, reasoning_effort, max_tokens=32000):
    cfg = AppConfig()
    cfg.llm = LLMConfig(provider="anthropic", model_id=model_id,
                        reasoning_effort=reasoning_effort, max_tokens=max_tokens)
    return cfg


def test_opus_gets_adaptive_thinking_and_effort(captured_anthropic):
    decision, raw = client_mod._call_anthropic(
        "sys", "user", _cfg("claude-opus-4-8", "high"))
    assert captured_anthropic["thinking"] == {"type": "adaptive"}
    assert captured_anthropic["output_config"] == {"effort": "high"}
    assert captured_anthropic["max_tokens"] == 32000        # streamed, thinking shares the budget
    assert "temperature" not in captured_anthropic          # Claude 4+ is temperature-free
    # Text is extracted from behind the leading thinking block
    assert decision.actions[0].ticker == "AAA.ST"
    assert raw == _VALID_JSON


def test_effort_respects_config_level(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-4-8", "medium"))
    assert captured_anthropic["output_config"] == {"effort": "medium"}


def test_no_reasoning_when_unset(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-4-8", None))
    assert "thinking" not in captured_anthropic
    assert "output_config" not in captured_anthropic


def test_effort_skipped_for_non_supporting_model(captured_anthropic):
    # Haiku 4.5 does not support the effort parameter — must not be sent.
    client_mod._call_anthropic("sys", "user", _cfg("claude-haiku-4-5", "high"))
    assert "thinking" not in captured_anthropic
    assert "output_config" not in captured_anthropic


# ── Current models: parameters follow family and version ─────────────────────

@pytest.mark.parametrize("model,sampling,effort", [
    ("claude-opus-5-5", False, True),
    ("claude-opus-5", False, True),
    ("claude-opus-4-8", False, True),
    ("claude-opus-4-5-20251101", False, False),
    ("claude-sonnet-5-5", False, True),
    ("claude-sonnet-5", False, True),
    ("claude-sonnet-4-6", False, True),
    ("claude-haiku-5-5", False, True),
    ("claude-haiku-4-5", False, False),
    ("claude-fable-5-2", False, True),
    ("claude-mythos-5-2", False, True),
    ("claude-3-5-sonnet-20241022", True, False),
])
def test_capabilities_are_read_from_the_model_id(model, sampling, effort):
    assert client_mod.claude_capabilities(model) == {"sampling": sampling, "effort": effort}


def test_opus_5_5_gets_its_effort_set_and_no_temperature(captured_anthropic):
    """It defaults to medium; the fund's 'high' must be sent, and sampling is a 400."""
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-5-5", "high"))
    assert captured_anthropic["output_config"] == {"effort": "high"}
    assert captured_anthropic["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured_anthropic


def test_sonnet_5_is_no_longer_sent_a_temperature(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-sonnet-5", "high"))
    assert "temperature" not in captured_anthropic


def test_an_openai_only_effort_falls_back_to_high(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-5-5", "minimal"))
    assert captured_anthropic["output_config"] == {"effort": "high"}


def test_models_that_can_decline_run_with_a_refusal_fallback(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-5-5", "high"))
    assert captured_anthropic["_endpoint"] == "beta"
    assert captured_anthropic["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured_anthropic["fallbacks"] == [{"model": "claude-opus-4-8"}]


def test_older_models_keep_the_stable_endpoint(captured_anthropic):
    client_mod._call_anthropic("sys", "user", _cfg("claude-opus-4-8", "high"))
    assert captured_anthropic["_endpoint"] == "stable"
    assert "fallbacks" not in captured_anthropic


def test_a_refusal_is_an_error_not_a_parse_failure(captured_anthropic):
    final = captured_anthropic["_final"]
    final.stop_reason, final.content = "refusal", []
    final.stop_details = types.SimpleNamespace(category="cyber")
    with pytest.raises(client_mod.LLMError, match="declined.*cyber"):
        client_mod._call_anthropic("sys", "user", _cfg("claude-opus-5-5", "high"))


def test_a_sample_another_model_answered_is_named(captured_anthropic, capsys):
    captured_anthropic["_final"].model = "claude-opus-4-8"
    decision, _ = client_mod._call_anthropic("sys", "user", _cfg("claude-opus-5-5", "high"))
    assert decision.actions[0].ticker == "AAA.ST"
    assert "claude-opus-4-8 answered" in capsys.readouterr().out
