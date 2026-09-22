"""Reversible historical context packing; never summarize away evidence or numbers."""
from collections import Counter
import copy
import json

VERSION = "exact_lines_v1"
FIELDS = ("macro", "portfolio_state", "risk_limits", "universe", "learnings")
PREFIX = ('The following JSON encodes the complete decision-time context. '
          'Expand each integer in lines using the zero-based shared array; strings are literal. '
          'Concatenate lines in order. References retain every occurrence and its attribution.\n')


def pack(text):
    lines = text.splitlines(keepends=True)
    counts = Counter(lines)
    shared = [line for line, count in counts.items() if count > 1 and len(line.encode()) >= 80]
    if not shared:
        return text
    indices = {line: i for i, line in enumerate(shared)}
    encoded = PREFIX + json.dumps({'shared': shared, 'lines': [indices.get(line, line) for line in lines]},
                                  ensure_ascii=False, separators=(',', ':'))
    # Never increase request size merely to use the compact representation.
    return encoded if len(encoded.encode()) < len(text.encode()) else text


def unpack(text):
    if not text.startswith(PREFIX):
        return text
    data = json.loads(text[len(PREFIX):])
    return ''.join(data['shared'][line] if isinstance(line, int) else line for line in data['lines'])


def raw_input(case):
    return '\n\n'.join(case['fields'][key] for key in FIELDS)


def prepare(plan, mode):
    if mode not in ('full', 'compact', 'compare'):
        raise ValueError('Unknown context mode')
    result = copy.deepcopy(plan)
    result['context_mode'] = mode
    result['context_version'] = VERSION
    for case in result['cases']:
        full = raw_input(case)
        compact = pack(full)
        if unpack(compact) != full:
            raise ValueError('Context reconstruction failed')
        case['compact_input'] = compact
        if mode == 'compact':
            case['search_input'] = compact
    return result


def profile(plan):
    fields = {key: sum(len(case['fields'][key].encode()) for case in plan['cases'])
              for key in ('mandate', *FIELDS)}
    full = sum(len(raw_input(case).encode()) for case in plan['cases'])
    compact = sum(len(pack(raw_input(case)).encode()) for case in plan['cases'])
    return {'field_bytes': fields, 'full_bytes': full, 'compact_bytes': compact,
            'saved_bytes': full - compact}


def compare_decisions(full, compact):
    """Expose changes for review, without claiming alpha or risk equivalence."""
    def actions(decision):
        return sorted((a.model_dump(mode='json') for a in decision.actions),
                      key=lambda a: json.dumps(a, sort_keys=True))
    before, after = actions(full), actions(compact)
    return {'actions_changed': before != after,
            'cash_target_changed': full.cash_target_pct != compact.cash_target_pct,
            'full_actions': before, 'compact_actions': after,
            'full_cash_target_pct': full.cash_target_pct,
            'compact_cash_target_pct': compact.cash_target_pct,
            'risk_equivalence': 'not established; review against frozen risk_limits',
            'note': 'One paired sample; differences may also reflect model sampling.'}
