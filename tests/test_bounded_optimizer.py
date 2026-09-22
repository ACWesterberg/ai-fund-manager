from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from fundmgr.config import AppConfig
from fundmgr.engine import bounded_optimizer as bo
from fundmgr.engine.client import LLMError
from fundmgr.engine.optimizer import guidance_versions
from fundmgr.engine.schema import Action, DecisionRun


@pytest.fixture
def search(tmp_path, monkeypatch):
    cfg = AppConfig(db_path=tmp_path / "fund.db", mandate_path=tmp_path / "mandate.md")
    cfg.mandate_path.write_text("Invest prudently")
    cfg.optimizer.compiled_dir = tmp_path / "compiled"
    examples = [{"run_id": f"2026-01-{i+1:02}", "source": "fund", "mandate": "Invest prudently",
                 "macro": "Market", "portfolio_state": f"Cash {100000+i}", "risk_limits": "Limit 18%",
                 "universe": "A and B", "learnings": "Check evidence", "ticker_alphas": {"A": -2, "B": 2}}
                for i in range(15)]
    plan = bo.make_plan(cfg, examples)
    path = bo.checkpoint_path(cfg, plan)
    calls = []
    def provider(system, user, task_cfg, schema, max_retries, on_usage=None):
        calls.append((system, user, task_cfg, schema, max_retries))
        assert max_retries == 0
        if on_usage:
            on_usage({"input_tokens": 100, "output_tokens": 20})
        if schema is bo.Proposal:
            result = bo.Proposal(instructions="Prefer B on corroborated evidence")
        else:
            ticker = "B" if "Prefer B" in system else "A"
            result = DecisionRun(run_id="sample", market_summary="Test", cash_target_pct=90,
                actions=[Action(ticker=ticker, side="buy", target_weight_pct=10,
                                sek_estimate=10000, confidence=.8, thesis="Evidence")])
        return result, result.model_dump_json()
    monkeypatch.setattr(bo, "call_llm", provider)
    return cfg, plan, path, calls, provider


def test_seven_call_instruction_search_stages_only_complete_winner(search):
    cfg, plan, path, calls, _ = search
    assert bo.run_search(cfg, plan, path)
    assert len(calls) == 7
    assert all(call[2].llm.max_tokens == 2048 for call in calls)
    assert all(call[2].llm.reasoning_effort == "low" for call in calls)
    assert cfg.llm.max_tokens == 4096  # Live config never changed.
    state = bo._load(path)
    assert state["status"] == "complete"
    assert guidance_versions(cfg)["current"] is None
    candidate = guidance_versions(cfg)["candidates"][0]
    assert candidate["search_method"] == bo.VERSION
    assert candidate["status"] == "pending_evaluation"
    assert bo.run_search(cfg, plan, path) and len(calls) == 7
    assert len(guidance_versions(cfg)["candidates"]) == 1


def test_proposal_never_sees_validation_context_or_labels(search):
    _, plan, _, _, _ = search
    for case in plan["cases"]:
        assert case["run_id"] not in plan["proposal_user"]
    assert "ticker_alphas" not in bo.task_input(plan["cases"][0])


@pytest.mark.parametrize("budget", ["max_calls", "max_total_tokens"])
def test_unaffordable_plan_is_rejected_before_first_call(search, budget):
    cfg, plan, path, calls, _ = search
    setattr(cfg.optimizer, budget, 1)
    with pytest.raises(bo.SearchStopped, match="exceeds budget"):
        bo.run_search(cfg, plan, path)
    assert not calls and not path.exists()


def test_quota_failure_stops_without_zero_score_and_resume_reuses_successes(search, monkeypatch):
    cfg, plan, path, calls, provider = search
    def fail(system, user, *a, **kw):
        if len(calls) == 2:
            raise LLMError("You have no credits remaining")
        return provider(system, user, *a, **kw)
    monkeypatch.setattr(bo, "call_llm", fail)
    with pytest.raises(LLMError, match="credits"):
        bo.run_search(cfg, plan, path)
    state = bo._load(path)
    assert len(state["results"]) == 2 and len(state["attempts"]) == 3
    assert "scores" not in state and not guidance_versions(cfg)["candidates"]
    with pytest.raises(bo.SearchStopped, match="Unfinished"):
        bo.run_search(cfg, plan, path)
    monkeypatch.setattr(bo, "call_llm", provider)
    with pytest.raises(bo.SearchStopped, match="retry-failed"):
        bo.run_search(cfg, plan, path, resume=True)
    cfg.optimizer.max_calls = 8  # Explicitly budget one additional attempt.
    assert bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    assert len(calls) == 7  # Two completed calls reused, five remaining calls made.
    assert len(bo._load(path)["attempts"]) == 8


def test_resume_never_resets_cumulative_budget(search, monkeypatch):
    cfg, plan, path, calls, provider = search
    def fail(*a, **kw):
        if len(calls) == 1:
            raise LLMError("timeout")
        return provider(*a, **kw)
    monkeypatch.setattr(bo, "call_llm", fail)
    with pytest.raises(LLMError):
        bo.run_search(cfg, plan, path)
    cfg.optimizer.max_calls = 2
    monkeypatch.setattr(bo, "call_llm", provider)
    with pytest.raises(bo.SearchStopped, match="Budget exhausted"):
        bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    assert len(calls) == 1 and len(bo._load(path)["attempts"]) == 2


def test_ctrl_c_saves_pending_request_and_completed_results(search, monkeypatch):
    cfg, plan, path, calls, provider = search
    def stop(*a, **kw):
        if calls:
            raise KeyboardInterrupt()
        return provider(*a, **kw)
    monkeypatch.setattr(bo, "call_llm", stop)
    with pytest.raises(KeyboardInterrupt):
        bo.run_search(cfg, plan, path)
    state = bo._load(path)
    assert state["status"] == "stopped" and "proposal" in state["results"]
    assert state["attempts"][-1]["status"] == "failed"


def test_settings_changes_and_corrupt_checkpoint_reject_without_calls(search):
    cfg, plan, path, calls, _ = search
    cfg.optimizer.max_output_tokens = 4096
    with pytest.raises(ValueError, match="settings changed"):
        bo.run_search(cfg, plan, path)
    cfg.optimizer.max_output_tokens = 2048
    path.parent.mkdir(parents=True)
    bo._save(path, {"status": "stopped", "plan": plan, "attempts": [], "results": {}})
    value = json.loads(path.read_text())
    value["attempts"] = [{"reserved_tokens": 0}]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        bo.run_search(cfg, plan, path, resume=True)
    assert not calls


def test_no_improvement_creates_no_candidate(search, monkeypatch):
    cfg, plan, path, calls, provider = search
    def equal(system, user, *a, **kw):
        return provider(system.replace("Prefer B", "Prefer A"), user, *a, **kw)
    monkeypatch.setattr(bo, "call_llm", equal)
    assert not bo.run_search(cfg, plan, path)
    assert bo._load(path)["status"] == "complete"
    assert not guidance_versions(cfg)["candidates"]


def test_crash_after_publication_does_not_duplicate_candidate(search):
    cfg, plan, path, calls, _ = search
    bo.run_search(cfg, plan, path)
    state = bo._load(path)
    state["status"] = "stopped"
    state.pop("candidate_path")
    bo._save(path, state)
    assert bo.run_search(cfg, plan, path, resume=True)
    assert len(calls) == 7 and len(guidance_versions(cfg)["candidates"]) == 1
    assert Path(bo._load(path)["candidate_path"]).exists()


def test_cli_dry_run_discloses_budget_and_makes_no_calls(search, monkeypatch):
    import fundmgr.cli as commands
    from fundmgr.engine import optimizer
    cfg, plan, path, calls, _ = search
    cfg.optimizer.min_examples = 1
    monkeypatch.setattr(commands, "_get_store", lambda: (cfg, SimpleNamespace(get_evaluated_outcomes=lambda: [None]*30)))
    monkeypatch.setattr(optimizer, "build_pooled_trainset", lambda cfg: [None]*5)
    monkeypatch.setattr(bo, "make_plan", lambda *a: plan)
    result = CliRunner().invoke(commands.cli, ["optimize", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "7 planned calls" in result.output and "2048 tokens/call" in result.output
    assert "not a dollar estimate" in result.output
    assert not calls and not path.exists()


def test_openai_transport_disables_retries_and_honors_small_output_cap(monkeypatch):
    import openai
    from fundmgr.engine.client import _call_openai
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    constructor = MagicMock()
    constructor.return_value.beta.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            parsed=bo.Proposal(instructions="test"), content='{"instructions":"test"}'))])
    monkeypatch.setattr(openai, "OpenAI", constructor)
    cfg = AppConfig()
    cfg.llm.model_id = "gpt-5.6-sol"
    cfg.llm.max_tokens = 2048
    cfg.llm.reasoning_effort = "low"
    _call_openai("system", "user", cfg, bo.Proposal, max_retries=0)
    assert constructor.call_args.kwargs["max_retries"] == 0
    args = constructor.return_value.beta.chat.completions.parse.call_args.kwargs
    assert args["max_completion_tokens"] == 2048 and args["reasoning_effort"] == "low"
    constructor.return_value.beta.chat.completions.parse.return_value.choices[0].finish_reason = "length"
    with pytest.raises(LLMError, match="output token limit"):
        _call_openai("system", "user", cfg, bo.Proposal, max_retries=0)


def test_anthropic_transport_disables_retries_and_rejects_truncation(monkeypatch):
    import sys
    from fundmgr.engine.client import _call_anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"instructions":"test"}')])
    constructor = MagicMock()
    constructor.return_value.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = response
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=constructor))
    cfg = AppConfig()
    cfg.llm.provider = "anthropic"
    cfg.llm.model_id = "claude-opus-4-8"
    cfg.llm.max_tokens = 2048
    cfg.llm.reasoning_effort = "low"
    _call_anthropic("system", "user", cfg, bo.Proposal, max_retries=0)
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert constructor.return_value.messages.stream.call_args.kwargs["max_tokens"] == 2048
    response.stop_reason = "max_tokens"
    with pytest.raises(LLMError, match="output token limit"):
        _call_anthropic("system", "user", cfg, bo.Proposal, max_retries=0)


def test_cli_failure_is_nonzero_and_keeps_checkpoint(search, monkeypatch):
    import fundmgr.cli as commands
    from fundmgr.engine import optimizer
    cfg, plan, path, calls, _ = search
    cfg.optimizer.min_examples = 1
    monkeypatch.setattr(commands, "_get_store", lambda: (cfg, SimpleNamespace(get_evaluated_outcomes=lambda: [None]*30)))
    monkeypatch.setattr(optimizer, "build_pooled_trainset", lambda cfg: [None]*5)
    monkeypatch.setattr(bo, "make_plan", lambda *a: plan)
    def fail(*a, **kw):
        raise LLMError("insufficient_quota")
    monkeypatch.setattr(bo, "call_llm", fail)
    result = CliRunner().invoke(commands.cli, ["optimize"])
    assert result.exit_code != 0 and "insufficient_quota" in result.output
    assert bo._load(path)["status"] == "stopped"
    assert not guidance_versions(cfg)["candidates"]


def test_lock_blocks_concurrent_search_before_calls(search):
    import fcntl
    cfg, plan, path, calls, _ = search
    path.parent.mkdir(parents=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(bo.SearchStopped, match="already running"):
            bo.run_search(cfg, plan, path)
    assert not calls


def test_token_budget_remains_spent_on_resume(search, monkeypatch):
    cfg, plan, path, calls, provider = search
    def stop(*a, **kw):
        if calls:
            raise LLMError("timeout")
        return provider(*a, **kw)
    monkeypatch.setattr(bo, "call_llm", stop)
    with pytest.raises(LLMError):
        bo.run_search(cfg, plan, path)
    state = bo._load(path)
    cfg.optimizer.max_total_tokens = sum(a["reserved_tokens"] for a in state["attempts"])
    monkeypatch.setattr(bo, "call_llm", provider)
    with pytest.raises(bo.SearchStopped, match="Budget exhausted"):
        bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    assert len(calls) == 1


def test_cross_search_cache_and_usage_are_not_double_counted(search):
    import copy
    from fundmgr.engine.research_costs import usage_report
    cfg, plan, path, calls, _ = search
    bo.run_search(cfg, plan, path)
    second = copy.deepcopy(plan)
    second['proposal_user'] += '\nConsider another proposal.'
    second_path = bo.checkpoint_path(cfg, second)
    assert bo.run_search(cfg, second, second_path, force_search=True)
    assert len(calls) == 8  # A new proposal; all six exact evaluations reused.
    report = usage_report(cfg.optimizer.compiled_dir / 'searches')
    assert report['attempts'] == 8 and report['local_cache_hits'] == 6
    assert sum(v['input_tokens'] for v in report['by_model'].values()) == 800
    assert report['unknown_usage_attempts'] == 0
    third = copy.deepcopy(second)
    third['cases'][0]['fields']['portfolio_state'] = 'Changed portfolio'
    bo.run_search(cfg, third, bo.checkpoint_path(cfg, third), force_search=True)
    assert len(calls) == 11  # Proposal and the two changed evaluation requests.


def test_new_evidence_gate_requires_own_distinct_dates(search):
    import copy
    from fundmgr.engine.research_costs import evidence_gate, evidence_periods
    cfg, plan, path, calls, _ = search
    assert evidence_gate(cfg, plan)['eligible']
    bo.run_search(cfg, plan, path)
    second = copy.deepcopy(plan)
    second['proposal_user'] += 'new'
    assert not bo.run_search(cfg, second, bo.checkpoint_path(cfg, second))
    assert len(calls) == 7
    assert not evidence_gate(cfg, second)['eligible']
    assert evidence_periods([{'source': 'other', 'run_id': '2026-02-01'},
                            {'source': 'fund', 'run_id': '2026-02-02T12:00'},
                            {'source': 'fund', 'run_id': '2026-02-02T13:00'}], 'fund') == ['2026-02-02']
    second['evidence_periods'] += ['2026-02-01', '2026-02-02', '2026-02-03']
    assert evidence_gate(cfg, second)['eligible']


def test_usage_survives_failed_parsing_and_missing_usage_is_unknown(search, monkeypatch):
    from fundmgr.engine.research_costs import usage_report, evidence_gate
    cfg, plan, path, _, _ = search
    def fail(*args, on_usage, **kwargs):
        on_usage({'input_tokens': 120, 'output_tokens': 50})
        raise LLMError('invalid output')
    monkeypatch.setattr(bo, 'call_llm', fail)
    with pytest.raises(LLMError):
        bo.run_search(cfg, plan, path)
    assert not evidence_gate(cfg, plan)['eligible']
    report = usage_report(cfg.optimizer.compiled_dir / 'searches')
    assert sum(v['output_tokens'] for v in report['by_model'].values()) == 50
    saved = bo._load(path)
    saved['attempts'][0].pop('usage')
    bo._save(path, saved)
    assert usage_report(cfg.optimizer.compiled_dir / 'searches')['unknown_usage_attempts'] == 1


@pytest.mark.parametrize('provider,raw,expected', [
    ('openai', {'prompt_tokens': 100, 'completion_tokens': 30,
                'prompt_tokens_details': {'cached_tokens': 80},
                'completion_tokens_details': {'reasoning_tokens': 20}}, (100, 30, 80, 20)),
    ('anthropic', {'input_tokens': 10, 'output_tokens': 30,
                   'cache_read_input_tokens': 80, 'cache_creation_input_tokens': 10}, (100, 30, 80, None)),
])
def test_provider_usage_normalization(provider, raw, expected):
    from fundmgr.engine.client import _report_usage
    reported = []
    _report_usage(SimpleNamespace(usage=SimpleNamespace(**raw)), provider, reported.append)
    assert tuple(reported[0][k] for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens')) == expected
    _report_usage(SimpleNamespace(), provider, reported.append)
    assert reported[-1] is None


def test_usage_cli_is_offline(search, monkeypatch):
    import fundmgr.cli as commands
    cfg, plan, path, calls, _ = search
    bo.run_search(cfg, plan, path)
    monkeypatch.setattr(commands, 'load_config', lambda: cfg)
    result = CliRunner().invoke(commands.cli, ['optimizer-usage'])
    assert result.exit_code == 0, result.output
    assert '700 input' in result.output and '140 output' in result.output
    assert len(calls) == 7


def test_changed_model_does_not_reuse_evaluations(search):
    import copy
    cfg, plan, path, calls, _ = search
    bo.run_search(cfg, plan, path)
    cfg.llm.model_id = 'different-model'
    second = copy.deepcopy(plan)
    second['identity'] = bo.identity(cfg)
    bo.run_search(cfg, second, bo.checkpoint_path(cfg, second), force_search=True)
    assert len(calls) == 14


def test_corrupt_cache_is_rejected_before_evaluation_call(search):
    import copy
    cfg, plan, path, calls, _ = search
    bo.run_search(cfg, plan, path)
    for cache in (cfg.optimizer.compiled_dir / 'request_cache').glob('*.json'):
        value = json.loads(cache.read_text())
        value['raw'] = 'tampered'
        cache.write_text(json.dumps(value))
    second = copy.deepcopy(plan)
    second['proposal_user'] += 'new'
    with pytest.raises(ValueError, match='checksum'):
        bo.run_search(cfg, second, bo.checkpoint_path(cfg, second), force_search=True)
    assert len(calls) == 8
