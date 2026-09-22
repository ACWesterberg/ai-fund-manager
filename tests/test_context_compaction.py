import pytest

from fundmgr.engine.context_compaction import pack, unpack, prepare, profile, raw_input


def test_repeated_evidence_round_trips_with_numbers_attribution_and_order():
    news = 'News dated 2026-01-01: revenue -12.50%, debt SEK 150000; uncertainty remains. ' * 4 + '\n'
    text = '## Universe\n' + ''.join(f'Ticker {ticker}; price 123.450; eligible\n{news}'
                                       for ticker in ['A', 'B', 'C', 'D', 'E'])
    encoded = pack(text)
    assert len(encoded.encode()) < len(text.encode())
    assert unpack(encoded) == text
    assert news.strip() in encoded


@pytest.mark.parametrize('text', ['', 'Unique evidence\n\nRisk: 18%\n', 'Øresund 😃\r\n' * 3,
                                  '\tSpaces  123.450\nDo not buy A\n'])
def test_unique_or_small_context_is_unchanged(text):
    assert pack(text) == text
    assert unpack(pack(text)) == text


def test_prepare_preserves_fields_and_keeps_labels_out():
    fields = {key: ('Historic ' + key + '\n') * 3 for key in
              ('mandate', 'macro', 'portfolio_state', 'risk_limits', 'universe', 'learnings')}
    plan = {'cases': [{'fields': fields, 'ticker_alphas': {'SECRET_FUTURE': 987654321}}]}
    compact = prepare(plan, 'compact')
    assert compact['cases'][0]['fields'] == fields
    assert 'search_input' not in plan['cases'][0]
    assert 'SECRET_FUTURE' not in compact['cases'][0]['compact_input']
    assert unpack(compact['cases'][0]['search_input']) == raw_input(plan['cases'][0])
    assert profile(compact)['saved_bytes'] >= 0


def test_unique_asset_values_share_metric_templates_without_losing_any_ticker():
    from fundmgr.engine.context_compaction import ASSET_PREFIX
    text = '## Universe\n'
    for i in range(100):
        text += (f'★ [TICKER{i}.ST] Company {i}\n'
                 f'  Price: {123+i}.45  (as of 2026-01-01)  [FX: SEK]  mktcap {2000+i}M\n'
                 f'  Returns: 1d +{i}.1%, 5d -{i}.2%, 20d +{i}.3%, 60d +{i}.4%, 52wH -{i}.5%\n'
                 f'  Technical: vol {i}%, RSI {i}, ▲ MA50, ▼ MA200, β 1.20\n'
                 f'  Calendar: earnings in {i}d, ex-div in {i+5}d, yield 3.4%\n'
                 f'    - 2026-01-01 [NEG] Unique uncertain report {i}\n'
                 f'      Contradictory evidence {i}. Do not infer certainty.\n'
                 f'  ⚠ DATA STALE ({i}d old)\n')
    compact = pack(text)
    assert compact.startswith(ASSET_PREFIX)
    assert unpack(compact) == text
    assert len(compact.encode()) < len(text.encode()) * .9
    for i in range(100):
        assert f'TICKER{i}.ST' in compact
        assert f'Contradictory evidence {i}.' in compact
