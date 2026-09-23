from datetime import date
from types import SimpleNamespace

import pytest

from fundmgr.engine import optimizer_estimate as oe


@pytest.fixture
def plan():
    return {'identity': {'llm': {'provider': 'openai', 'model_id': 'gpt-5.6-sol'},
                        'prompt_model': 'gpt-5.6-sol', 'guidance': 'Existing guidance', 'output_tokens': 2048},
            'proposal_system': 'Learn from past decisions', 'proposal_user': 'Training summaries only',
            'cases': [{'fields': {key: key + ' text ÅÄÖ 123.45' for key in
                                 ('mandate', 'macro', 'portfolio_state', 'risk_limits', 'universe', 'learnings')},
                       'compact_input': 'compact'} for _ in range(3)]}


@pytest.fixture
def counter(monkeypatch):
    monkeypatch.setattr(oe, 'encoding_for', lambda model: SimpleNamespace(name='test'))
    monkeypatch.setattr(oe, 'count_input', lambda *args: 1000)


def test_batch_discount_only_applies_to_six_evaluations(plan, counter):
    direct = oe.estimate(plan, today=oe.PRICE_DATE)
    plan['execution'] = 'batch'
    batch = oe.estimate(plan, today=oe.PRICE_DATE)
    assert direct['input_low'] == batch['input_low'] == 7000
    assert batch['input_high'] == 7000 + 3 * 16000
    assert batch['output_max'] == 7 * 2048
    assert batch['groups'][0]['usd_high'] == direct['groups'][0]['usd_high']
    assert batch['groups'][1]['usd_high'] == direct['groups'][1]['usd_high'] / 2
    assert batch['usd_high'] == pytest.approx(.004 + .04096 + (54000*.000004 + 12288*.00002)/2)


def test_saved_candidate_removes_unknown_allowance(plan, counter):
    saved = {'results': {'proposal': {'parsed': {'instructions': 'New known guidance'}}}}
    report = oe.estimate(plan, saved, today=oe.PRICE_DATE)
    assert report['input_low'] == report['input_high'] == 7000
    assert 'not remaining resume cost' in '\n'.join(oe.describe(report))


def test_compare_has_no_proposal_and_no_unknown_candidate(plan, counter):
    plan['context_mode'] = 'compare'
    plan['execution'] = 'batch'
    report = oe.estimate(plan, today=oe.PRICE_DATE)
    assert len(report['groups']) == 1
    assert report['input_low'] == report['input_high'] == 6000
    assert report['output_max'] == 6 * 2048


def test_long_context_pricing_is_per_request(plan, counter, monkeypatch):
    monkeypatch.setattr(oe, 'count_input', lambda *args: 273000)
    report = oe.estimate(plan, {'results': {'proposal': {'parsed': {'instructions': 'known'}}}}, today=oe.PRICE_DATE)
    assert report['usd_high'] == pytest.approx(7 * (273000*8 + 2048*30)/1e6)


@pytest.mark.parametrize('change', ['claude', 'unknown_model', 'unknown_prompt_model', 'stale', 'missing_tokenizer'])
def test_unavailable_is_not_a_zero_or_partial_quote(plan, counter, monkeypatch, change):
    today = oe.PRICE_DATE
    if change == 'claude':
        plan['identity']['llm']['provider'] = 'anthropic'
    elif change == 'unknown_model':
        plan['identity']['llm']['model_id'] = 'unknown'
    elif change == 'unknown_prompt_model':
        plan['identity']['prompt_model'] = 'unknown'
    elif change == 'stale':
        today = date(2026, 11, 1)
    else:
        def missing(*args):
            raise ImportError()
        monkeypatch.setattr(oe, 'encoding_for', missing)
    report = oe.estimate(plan, today=today)
    assert not report['available'] and 'usd_high' not in report
    assert 'unavailable' in oe.describe(report)[0]


def test_schema_and_model_tokenizer_count_real_unicode_and_literal_special_tokens(plan):
    # No provider client or credentials required. Public tokenizer vocabulary may
    # be downloaded once by tiktoken, then all prompt text stays local.
    import hashlib
    import os
    import tempfile
    from pathlib import Path
    import tiktoken
    cache = Path(os.environ.get('TIKTOKEN_CACHE_DIR', os.environ.get('DATA_GYM_CACHE_DIR',
                 str(Path(tempfile.gettempdir()) / 'data-gym-cache'))))
    vocabulary = 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
    if not (cache / hashlib.sha1(vocabulary.encode()).hexdigest()).exists():
        pytest.skip('Public tokenizer vocabulary not cached; tests do not download it')
    encoding = tiktoken.encoding_for_model('gpt-5.6-sol')
    assert len(encoding.encode('Hello world', disallowed_special=())) == 2
    from fundmgr.engine.bounded_optimizer import Proposal
    count = oe.count_input(encoding, 'ÅÄÖ <|endoftext|>', 'Prices 123.45', Proposal)
    assert count > 32
    report = oe.estimate(plan, today=oe.PRICE_DATE)
    assert report['available'] and report['input_low'] > 0
    assert report['groups'][0]['encoding'] == encoding.name


def test_dry_run_displays_estimate_without_changing_limits_or_calling_model(plan, counter, tmp_path, monkeypatch):
    from click.testing import CliRunner
    import fundmgr.cli as commands
    from fundmgr.config import AppConfig
    from fundmgr.engine import bounded_optimizer as bo, optimizer
    cfg = AppConfig(db_path=tmp_path / 'fund.db')
    cfg.optimizer.compiled_dir = tmp_path / 'compiled'
    cfg.optimizer.min_examples = 1
    plan['training_runs'] = []
    plan['evidence_periods'] = []
    report = oe.estimate(plan, today=oe.PRICE_DATE)
    monkeypatch.setattr(oe, 'estimate', lambda *args: report)
    monkeypatch.setattr(commands, '_get_store', lambda: (cfg, SimpleNamespace(get_evaluated_outcomes=lambda: [None]*30)))
    monkeypatch.setattr(optimizer, 'build_pooled_trainset', lambda cfg: [None]*25)
    monkeypatch.setattr(bo, 'make_plan', lambda *args: plan)
    def no_calls(*args, **kwargs):
        pytest.fail('Dry run attempted execution')
    monkeypatch.setattr(bo, 'run_search', no_calls)
    result = CliRunner().invoke(commands.cli, ['optimize', '--context-mode', 'full', '--execution', 'batch', '--dry-run'])
    assert result.exit_code == 0, result.output
    assert 'Estimated whole-plan cost: USD $' in result.output
    assert 'not remaining resume cost' in result.output
    assert cfg.optimizer.max_total_tokens == 200000
    assert not list((tmp_path / 'compiled').rglob('*.json'))
