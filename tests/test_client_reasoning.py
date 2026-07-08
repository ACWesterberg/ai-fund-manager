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

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            # Mimic Opus-with-thinking: a thinking block precedes the text block.
            return types.SimpleNamespace(content=[
                _Block("thinking", thinking="...reasoning..."),
                _Block("text", text=_VALID_JSON),
            ])

    class _Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return captured


def _cfg(model_id, reasoning_effort):
    cfg = AppConfig()
    cfg.llm = LLMConfig(provider="anthropic", model_id=model_id,
                        reasoning_effort=reasoning_effort, max_tokens=16000)
    return cfg


def test_opus_gets_adaptive_thinking_and_effort(captured_anthropic):
    decision, raw = client_mod._call_anthropic(
        "sys", "user", _cfg("claude-opus-4-8", "high"))
    assert captured_anthropic["thinking"] == {"type": "adaptive"}
    assert captured_anthropic["output_config"] == {"effort": "high"}
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
