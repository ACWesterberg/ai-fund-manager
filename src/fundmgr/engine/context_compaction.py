"""Reversible historical context packing; never summarize away evidence or numbers."""
from collections import Counter
import copy
import json
import re

VERSION = "asset_templates_v2"
FIELDS = ("macro", "portfolio_state", "risk_limits", "universe", "learnings")
PREFIX = ('The following JSON encodes the complete decision-time context. '
          'Expand each integer in lines using the zero-based shared array; strings are literal. '
          'Concatenate lines in order. References retain every occurrence and its attribution.\n')


def _pack_lines(text):
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


# Only factor rendered feature rows. Narrative, names, tickers and warnings stay
# literal. Captures are strings: never round, reformat or infer missing values.
FEATURE_ROW = re.compile(r"^  (?:Price|Returns|Technical|Valuation|Quality|Growth \(YoY\)|Analysts|Calendar|Volume|Sentiment \(FinBERT\)):")
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)*")
ASSET_PREFIX = ('Complete historical context in compact rows. Each string row is literal. '
                'Each array row [template_id, values...] expands by interleaving those values '
                'between the fragments of templates[template_id]. All values are exact text, '
                'including signs, dates and units; preserve row order and ticker attribution.\n')


def _pack_assets(text):
    rows, counts = [], Counter()
    for line in text.splitlines(keepends=True):
        if FEATURE_ROW.match(line):
            fragments = tuple(NUMBER.split(line))
            values = NUMBER.findall(line)
            if values:
                rows.append((fragments, values))
                counts[fragments] += 1
                continue
        rows.append(line)
    templates = [fragments for fragments, count in counts.items()
                 if count > 1 and sum(map(len, fragments)) > 24]
    if not templates:
        return text
    indices = {fragments: i for i, fragments in enumerate(templates)}
    output = []
    for row in rows:
        if isinstance(row, str):
            output.append(row)
        else:
            fragments, values = row
            if fragments in indices:
                output.append([indices[fragments], *values])
            else:
                output.append(_expand(fragments, values))
    # Coalesce literal blocks so narrative does not pay JSON overhead per line.
    merged = []
    for row in output:
        if isinstance(row, str) and merged and isinstance(merged[-1], str):
            merged[-1] += row
        else:
            merged.append(row)
    return ASSET_PREFIX + json.dumps({'templates': templates, 'rows': merged},
                                     ensure_ascii=False, separators=(',', ':'))


def _expand(fragments, values):
    if len(fragments) != len(values) + 1:
        raise ValueError('Malformed compact feature row')
    return ''.join(fragment + value for fragment, value in zip(fragments, values)) + fragments[-1]


def pack(text):
    candidates = [text, _pack_lines(text), _pack_assets(text)]
    return min(candidates, key=lambda candidate: len(candidate.encode()))


def unpack(text):
    if text.startswith(ASSET_PREFIX):
        data = json.loads(text[len(ASSET_PREFIX):])
        return "".join(row if isinstance(row, str) else _expand(data["templates"][row[0]], row[1:])
                       for row in data["rows"])
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
    compact = sum(len(case.get("compact_input", pack(raw_input(case))).encode()) for case in plan['cases'])
    feature_bytes = sum(len(line.encode()) for case in plan['cases']
                        for line in case['fields']['universe'].splitlines(keepends=True) if FEATURE_ROW.match(line))
    return {'universe_feature_bytes': feature_bytes,
            'universe_other_bytes': fields['universe'] - feature_bytes,
            'field_bytes': fields, 'full_bytes': full, 'compact_bytes': compact,
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
