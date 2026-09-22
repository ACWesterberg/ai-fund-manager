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
