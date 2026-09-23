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


def test_context_comparison_is_bounded_resumable_and_never_publishes(search):
    from fundmgr.engine.context_compaction import prepare
    from fundmgr.engine.research_costs import evidence_gate, usage_report
    cfg, plan, _, calls, _ = search
    repeated = ('Material evidence from source dated 2026-01-01: revenue down 15%; ' * 5) + '\n'
    for case in plan['cases']:
        case['fields']['universe'] = ''.join(f'Ticker {i}\n{repeated}' for i in range(8))
    plan = prepare(plan, 'compare')
    path = bo.checkpoint_path(cfg, plan)
    assert bo.plan_cost(plan)['planned_calls'] == 6
    assert not bo.run_search(cfg, plan, path)
    assert len(calls) == 6 and all(call[3] is DecisionRun for call in calls)
    state = bo._load(path)
    assert len(state['comparisons']) == 3
    assert not guidance_versions(cfg)['candidates']
    assert evidence_gate(cfg, plan)['first_search']
    assert usage_report(cfg.optimizer.compiled_dir / 'searches')['attempts'] == 6
    assert not bo.run_search(cfg, plan, path, resume=True)
    assert len(calls) == 6


def test_compact_search_preserves_original_and_costs_match_request(search):
    from fundmgr.engine.context_compaction import prepare, unpack
    cfg, plan, _, calls, _ = search
    for case in plan['cases']:
        case['fields']['universe'] = ('Exact dated evidence and risk uncertainty. ' * 10 + '\n') * 12
    compact = prepare(plan, 'compact')
    assert bo.plan_cost(compact)['reserved_token_estimate'] < bo.plan_cost(plan)['reserved_token_estimate']
    bo.run_search(cfg, compact, bo.checkpoint_path(cfg, compact))
    assert unpack(calls[1][1]) == bo.raw_task_input(plan['cases'][0])
    candidate = guidance_versions(cfg)['candidates'][0]
    assert candidate['context_mode'] == 'compact'


def test_context_cli_dry_run_and_resume_mode_guard(search, monkeypatch):
    import fundmgr.cli as commands
    from fundmgr.engine import optimizer
    cfg, plan, _, calls, _ = search
    cfg.optimizer.min_examples = 1
    monkeypatch.setattr(commands, '_get_store', lambda: (cfg, SimpleNamespace(get_evaluated_outcomes=lambda: [None]*30)))
    monkeypatch.setattr(optimizer, 'build_pooled_trainset', lambda cfg: [None]*5)
    monkeypatch.setattr(bo, 'make_plan', lambda *a: plan)
    result = CliRunner().invoke(commands.cli, ['optimize', '--context-mode', 'compare', '--dry-run'])
    assert result.exit_code == 0, result.output
    assert 'Context comparison: 6 planned calls' in result.output
    assert 'universe:' in result.output and not calls
    result = CliRunner().invoke(commands.cli, ['optimize', '--context-mode', 'compare'])
    assert result.exit_code == 0, result.output
    assert 'No guidance candidate created' in result.output
    path = next((cfg.optimizer.compiled_dir / 'searches' / 'fund').glob('*.json'))
    result = CliRunner().invoke(commands.cli, ['optimize', '--resume', str(path), '--context-mode', 'compact'])
    assert result.exit_code != 0 and 'context mode' in result.output


@pytest.fixture
def fake_batch(monkeypatch):
    from fundmgr.engine import optimizer_batch as batch
    client = MagicMock()
    submitted = []
    payloads = []
    def upload(**kw):
        payloads.append([json.loads(line) for line in kw['file'][1].decode().splitlines()])
        return SimpleNamespace(id=f'file-{len(payloads)}')
    def create(**kw):
        remote = SimpleNamespace(id=f'batch-{len(submitted)}', status='in_progress',
                                 input_file_id=kw['input_file_id'], metadata=kw['metadata'],
                                 output_file_id=f'output-{len(submitted)}', error_file_id=None)
        submitted.append(remote)
        return remote
    def content(file_id):
        index = int(file_id.split('-')[1])
        rows = []
        for request in reversed(payloads[index]):
            ticker = 'B' if 'Prefer B' in request['body']['messages'][0]['content'] else 'A'
            result = DecisionRun(run_id='sample', market_summary='Test', cash_target_pct=90,
                actions=[Action(ticker=ticker, side='buy', target_weight_pct=10,
                                sek_estimate=10000, confidence=.8, thesis='Evidence')])
            rows.append({'custom_id': request['custom_id'], 'response': {'status_code': 200,
                         'body': {'choices': [{'finish_reason': 'stop', 'message': {'content': result.model_dump_json()}}],
                                  'usage': {'prompt_tokens': 200, 'completion_tokens': 30}}}})
        return SimpleNamespace(text='\n'.join(json.dumps(row) for row in rows))
    client.files.create.side_effect = upload
    client.batches.create.side_effect = create
    client.batches.retrieve.side_effect = lambda id: submitted[int(id.split('-')[1])]
    client.files.content.side_effect = content
    monkeypatch.setattr(batch, 'batch_client', lambda: client)
    return client, submitted, payloads


def test_batch_submits_six_evaluations_then_resumes_without_rebilling(search, fake_batch):
    from fundmgr.engine.optimizer_batch import BatchPending
    from fundmgr.engine.research_costs import usage_report
    cfg, plan, _, calls, _ = search
    client, submitted, payloads = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending, match='submitted'):
        bo.run_search(cfg, plan, path)
    assert len(calls) == 1 and len(payloads[0]) == 6
    assert bo._load(path)['status'] == 'batch_pending'
    assert not guidance_versions(cfg)['candidates']
    with pytest.raises(BatchPending, match='in_progress'):
        bo.run_search(cfg, plan, path, resume=True)
    assert client.batches.create.call_count == 1
    submitted[0].status = 'completed'
    assert bo.run_search(cfg, plan, path, resume=True)
    assert len(calls) == 1 and len(bo._load(path)['attempts']) == 7
    report = usage_report(cfg.optimizer.compiled_dir / 'searches')
    assert sum(m['input_tokens'] for m in report['by_model'].values()) == 1300
    assert bo.run_search(cfg, plan, path, resume=True)
    assert client.batches.create.call_count == 1


def test_uncertain_batch_requires_matching_id_even_with_retry_flag(search, fake_batch):
    cfg, plan, _, calls, _ = search
    client, submitted, _ = fake_batch
    original = client.batches.create.side_effect
    def uncertain(**kw):
        original(**kw)
        raise TimeoutError('connection lost after acceptance')
    client.batches.create.side_effect = uncertain
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(bo.SearchStopped, match='uncertain'):
        bo.run_search(cfg, plan, path)
    with pytest.raises(bo.SearchStopped, match='Uncertain'):
        bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    assert client.batches.create.call_count == 1
    submitted[0].status = 'completed'
    submitted[0].metadata = {'wrong': 'metadata'}
    with pytest.raises(ValueError, match='does not match'):
        bo.run_search(cfg, plan, path, resume=True, batch_id='batch-0')
    submitted[0].metadata = bo._load(path)['batches'][0]['metadata']
    assert bo.run_search(cfg, plan, path, resume=True, batch_id='batch-0')
    assert len(calls) == 1


def test_batch_partial_errors_save_success_and_retry_only_failed(search, fake_batch):
    from fundmgr.engine.optimizer_batch import BatchPending
    cfg, plan, _, calls, _ = search
    client, submitted, payloads = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    original = client.files.content.side_effect
    def partial(file_id):
        rows = [json.loads(line) for line in original(file_id).text.splitlines()]
        rows[0] = {'custom_id': rows[0]['custom_id'], 'error': {'code': 'batch_expired'}}
        return SimpleNamespace(text='\n'.join(json.dumps(row) for row in rows))
    client.files.content.side_effect = partial
    submitted[0].status = 'expired'
    with pytest.raises(bo.SearchStopped, match='1 batch request'):
        bo.run_search(cfg, plan, path, resume=True)
    assert len(bo._load(path)['results']) == 6  # proposal plus five successful evaluations
    with pytest.raises(bo.SearchStopped, match='retry-failed'):
        bo.run_search(cfg, plan, path, resume=True)
    with pytest.raises(bo.SearchStopped, match='Budget exhausted'):
        bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    cfg.optimizer.max_calls = 8
    client.files.content.side_effect = original
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path, resume=True, retry_failed=True)
    assert len(payloads[1]) == 1
    submitted[1].status = 'completed'
    assert bo.run_search(cfg, plan, path, resume=True)
    assert len(calls) == 1


def test_unsupported_batch_provider_and_budget_fail_before_proposal(search, fake_batch):
    cfg, plan, _, calls, _ = search
    client, _, _ = fake_batch
    plan['execution'] = 'batch'
    cfg.optimizer.max_calls = 1
    with pytest.raises(bo.SearchStopped, match='exceeds budget'):
        bo.run_search(cfg, plan, bo.checkpoint_path(cfg, plan))
    cfg.llm.provider = 'anthropic'
    with pytest.raises(ValueError, match='OpenAI funds only'):
        bo.run_search(cfg, plan, bo.checkpoint_path(cfg, plan))
    assert not calls and not client.files.create.called


@pytest.mark.parametrize('problem', ['truncated', 'refused', 'invalid_json'])
def test_batch_invalid_response_retains_usage_and_never_scores_zero(search, fake_batch, problem):
    from fundmgr.engine.optimizer_batch import BatchPending
    cfg, plan, _, _, _ = search
    client, submitted, _ = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    original = client.files.content.side_effect
    def invalid(file_id):
        rows = [json.loads(line) for line in original(file_id).text.splitlines()]
        choice = rows[0]['response']['body']['choices'][0]
        if problem == 'truncated':
            choice['finish_reason'] = 'length'
        elif problem == 'refused':
            choice['message']['refusal'] = 'refused'
        else:
            choice['message']['content'] = 'invalid JSON'
        return SimpleNamespace(text='\n'.join(json.dumps(row) for row in rows))
    client.files.content.side_effect = invalid
    submitted[0].status = 'completed'
    with pytest.raises(bo.SearchStopped, match='1 batch request'):
        bo.run_search(cfg, plan, path, resume=True)
    state = bo._load(path)
    assert 'scores' not in state and not guidance_versions(cfg)['candidates']
    assert sum(a['usage']['input_tokens'] for a in state['attempts']) == 1300


def test_batch_network_failure_during_retrieval_never_resubmits(search, fake_batch):
    from fundmgr.engine.optimizer_batch import BatchPending
    cfg, plan, _, calls, _ = search
    client, _, _ = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    client.batches.retrieve.side_effect = TimeoutError('read timeout')
    with pytest.raises(LLMError, match='checkpoint preserved'):
        bo.run_search(cfg, plan, path, resume=True)
    assert client.batches.create.call_count == 1 and len(calls) == 1
    assert bo._load(path)['batches'][0]['id'] == 'batch-0'


def test_batch_cli_dry_run_submission_and_resume(search, fake_batch, monkeypatch):
    import fundmgr.cli as commands
    from fundmgr.engine import optimizer
    cfg, plan, _, calls, _ = search
    client, submitted, payloads = fake_batch
    cfg.llm.model_id = "gpt-5.6-sol"
    plan["identity"] = bo.identity(cfg)
    cfg.optimizer.min_examples = 1
    monkeypatch.setattr(commands, '_get_store', lambda: (cfg, SimpleNamespace(get_evaluated_outcomes=lambda: [None]*30)))
    monkeypatch.setattr(optimizer, 'build_pooled_trainset', lambda cfg: [None]*5)
    monkeypatch.setattr(bo, 'make_plan', lambda *a: dict(plan))
    result = CliRunner().invoke(commands.cli, ['optimize', '--execution', 'batch', '--dry-run'])
    assert result.exit_code == 0, result.output
    assert 'proposal (no universe)' in result.output
    assert not client.files.create.called and not calls
    result = CliRunner().invoke(commands.cli, ['optimize', '--execution', 'batch'])
    assert result.exit_code == 0 and 'submitted' in result.output, result.output
    body = payloads[0][0]['body']
    assert body['response_format']['json_schema']['strict']
    assert body['max_completion_tokens'] == 2048
    assert body['reasoning_effort'] == 'low'
    assert 'ticker_alphas' not in json.dumps(body)
    path = next((cfg.optimizer.compiled_dir / 'searches' / 'fund').glob('*.json'))
    submitted[0].status = 'completed'
    result = CliRunner().invoke(commands.cli, ['optimize', '--resume', str(path)])
    assert result.exit_code == 0 and 'Inactive candidate saved' in result.output, result.output


@pytest.mark.parametrize('reuse,expected_requests', [(True, 3), (False, 6)])
def test_batch_context_comparison_honors_reuse_setting(search, fake_batch, reuse, expected_requests):
    from fundmgr.engine.context_compaction import prepare
    from fundmgr.engine.optimizer_batch import BatchPending
    cfg, plan, _, calls, _ = search
    _, submitted, payloads = fake_batch
    cfg.optimizer.reuse_evaluations = reuse
    plan = prepare(plan, 'compare')  # Small inputs stay literal: each full/compact pair is identical.
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    assert len(payloads[0]) == expected_requests
    submitted[0].status = 'completed'
    assert not bo.run_search(cfg, plan, path, resume=True)
    state = bo._load(path)
    assert len(state['results']) == 6 and len(state['attempts']) == expected_requests
    assert len(state['comparisons']) == 3
    assert not calls and not guidance_versions(cfg)['candidates']


def test_watcher_collects_once_notifies_once_and_ignores_smaller_current_budget(search, fake_batch, monkeypatch):
    from fundmgr.engine.optimizer_batch import BatchPending
    from fundmgr.engine import optimizer_watch as watcher
    cfg, plan, _, calls, _ = search
    client, submitted, _ = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    notifications = []
    monkeypatch.setattr(watcher, 'send_telegram', lambda message, **kwargs: notifications.append(message) or True)
    assert watcher.watch(cfg)['pending'] == 1
    assert not notifications
    cfg.optimizer.max_total_tokens = 1  # Collection does not reserve new paid requests.
    cfg.optimizer.max_calls = 1
    submitted[0].status = 'completed'
    result = watcher.watch(cfg)
    assert result['complete'] == result['notified'] == 1
    assert 'candidate saved' in notifications[0]
    assert watcher.watch(cfg)['notified'] == 0
    assert len(notifications) == 1 and len(calls) == 1
    assert client.batches.create.call_count == 1


def test_watcher_retries_delivery_without_recollecting_or_paying(search, fake_batch, monkeypatch):
    from fundmgr.engine.optimizer_batch import BatchPending
    from fundmgr.engine import optimizer_watch as watcher
    cfg, plan, _, calls, _ = search
    client, submitted, _ = fake_batch
    plan['execution'] = 'batch'
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, bo.checkpoint_path(cfg, plan))
    submitted[0].status = 'completed'
    sender = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(watcher, 'send_telegram', sender)
    assert watcher.watch(cfg)['notification_failed'] == 1
    assert watcher.watch(cfg)['notified'] == 1
    assert watcher.watch(cfg)['notified'] == 0
    assert sender.call_count == 2 and client.batches.retrieve.call_count == 1
    assert len(calls) == 1


def test_watcher_failed_batch_never_retries_paid_work(search, fake_batch, monkeypatch):
    from fundmgr.engine.optimizer_batch import BatchPending
    from fundmgr.engine import optimizer_watch as watcher
    cfg, plan, _, calls, _ = search
    client, submitted, _ = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    submitted[0].status = 'failed'
    submitted[0].output_file_id = None
    sender = MagicMock(return_value=True)
    monkeypatch.setattr(watcher, 'send_telegram', sender)
    assert watcher.watch(cfg)['attention'] == 1
    assert watcher.watch(cfg)['attention'] == 1
    assert sender.call_count == 1 and client.batches.create.call_count == 1
    assert len(calls) == 1 and not guidance_versions(cfg)['candidates']


def test_collection_only_cannot_start_new_search_or_missing_proposal(search, fake_batch):
    from fundmgr.engine.optimizer_batch import BatchPending
    cfg, plan, _, calls, _ = search
    client, _, _ = fake_batch
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(ValueError, match='existing batch'):
        bo.run_search(cfg, plan, path, resume=True, collect_only=True)
    assert not calls
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    state = bo._load(path)
    state['results'].pop('proposal')
    bo._save(path, state)
    with pytest.raises(bo.SearchStopped, match='cannot generate'):
        bo.run_search(cfg, plan, path, resume=True, collect_only=True, retry_failed=True)
    assert len(calls) == 1 and client.batches.create.call_count == 1


def test_watcher_busy_fund_is_quiet(search, fake_batch, monkeypatch):
    import fcntl
    from fundmgr.engine.optimizer_batch import BatchPending
    from fundmgr.engine import optimizer_watch as watcher
    cfg, plan, _, _, _ = search
    plan['execution'] = 'batch'
    path = bo.checkpoint_path(cfg, plan)
    with pytest.raises(BatchPending):
        bo.run_search(cfg, plan, path)
    sender = MagicMock(return_value=True)
    monkeypatch.setattr(watcher, 'send_telegram', sender)
    with (path.parent / 'fund.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watcher.watch(cfg)['pending'] == 1
    sender.assert_not_called()
