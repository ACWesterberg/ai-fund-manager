"""Advisory token/dollar estimates, independent of durable budget reservations."""
from datetime import date
import json

from fundmgr.engine.client import _schema_hint
from fundmgr.engine.prompt import assemble_system_prompt
from fundmgr.engine.schema import DecisionRun

PRICE_DATE = date(2026, 9, 23)
# Reviewed monthly; refuse to present stale rates as a current quote.
PRICE_VALID_UNTIL = date(2026, 10, 23)
PRICE_SOURCE = 'https://developers.openai.com/api/docs/models/gpt-5.6-sol'
BATCH_SOURCE = 'https://developers.openai.com/api/docs/guides/batch'
PRICES = {'gpt-5.6-sol': (4.0, 20.0)}  # USD / million tokens, standard tier
FRAMING_ALLOWANCE = 32  # Chat/schema server serialization is not public/exact.
CANDIDATE_ALLOWANCE = 16000  # 4000 Unicode characters, at most four UTF-8 bytes each.


def encoding_for(model):
    import tiktoken
    # Never silently substitute an unrelated tokenizer for unknown models.
    return tiktoken.encoding_for_model(model)


def count_input(encoding, system, user, schema):
    from openai.lib._pydantic import to_strict_json_schema
    structured = {'type': 'json_schema', 'json_schema': {
        'name': schema.__name__, 'strict': True, 'schema': to_strict_json_schema(schema)}}
    parts = [system + _schema_hint(schema), user,
             json.dumps(structured, ensure_ascii=False, separators=(',', ':'))]
    return sum(len(encoding.encode(text, disallowed_special=())) for text in parts) + FRAMING_ALLOWANCE


def request_estimates(plan, saved=None):
    from fundmgr.engine.bounded_optimizer import Proposal, task_input, raw_task_input
    ident = plan['identity']
    model = ident['llm']['model_id']
    batch = plan.get('execution') == 'batch'
    if plan.get('context_mode') == 'compare':
        for case in plan['cases']:
            system = assemble_system_prompt(case['fields']['mandate'], ident['guidance'])
            for user in (raw_task_input(case), case['compact_input']):
                yield 'evaluations', model, batch, system, user, DecisionRun, 0
        return
    yield 'proposal', ident['prompt_model'], False, plan['proposal_system'], plan['proposal_user'], Proposal, 0
    candidate = (saved or {}).get('results', {}).get('proposal', {}).get('parsed', {}).get('instructions')
    for case in plan['cases']:
        for guidance, allowance in ((ident['guidance'], 0),
                                    (candidate or '', 0 if candidate is not None else CANDIDATE_ALLOWANCE)):
            # Count the guidance heading even before its contents are known.
            system = assemble_system_prompt(case['fields']['mandate'], guidance or (' ' if allowance else ''))
            yield 'evaluations', model, batch, system, task_input(case), DecisionRun, allowance


def estimate(plan, saved=None, *, today=None):
    today = today or date.today()
    if plan['identity']['llm']['provider'] != 'openai':
        return {'available': False, 'reason': 'Local token estimates currently support OpenAI only; no Claude tokenizer guessed.'}
    if not PRICE_DATE <= today <= PRICE_VALID_UNTIL:
        return {'available': False, 'reason': f'Pricing snapshot dated {PRICE_DATE} needs review before quoting dollars.'}
    groups, encodings = {}, {}
    try:
        for role, model, batch, system, user, schema, allowance in request_estimates(plan, saved):
            if model not in PRICES:
                return {'available': False, 'reason': f'No verified price for {model}; no partial total quoted.'}
            if model not in encodings:
                encodings[model] = encoding_for(model)
            encoding = encodings[model]
            lower = count_input(encoding, system, user, schema)
            upper = lower + allowance
            output = plan['identity']['output_tokens']
            rate_in, rate_out = PRICES[model]
            discount = .5 if batch else 1.0
            def cost(tokens, include_output):
                long = tokens > 272000
                return discount * (tokens * rate_in * (2 if long else 1) +
                                   (output * rate_out * (1.5 if long else 1) if include_output else 0)) / 1_000_000
            key = (role, model, batch)
            group = groups.setdefault(key, {'role': role, 'model': model, 'batch': batch,
                'encoding': encoding.name, 'calls': 0, 'input_low': 0, 'input_high': 0,
                'output_max': 0, 'usd_low': 0.0, 'usd_high': 0.0})
            for field, value in (('calls', 1), ('input_low', lower), ('input_high', upper),
                                 ('output_max', output), ('usd_low', cost(lower, False)),
                                 ('usd_high', cost(upper, True))):
                group[field] += value
    except ImportError:
        return {'available': False, 'reason': 'Install current dependencies (including tiktoken) to count tokens.'}
    except Exception as exc:
        # Advisory display must not break resume or relax budget enforcement.
        return {'available': False, 'reason': f'Tokenizer unavailable ({type(exc).__name__}); no byte-to-token guess used.'}
    rows = list(groups.values())
    return {'available': True, 'groups': rows, 'price_date': PRICE_DATE.isoformat(),
            'usd_low': sum(r['usd_low'] for r in rows), 'usd_high': sum(r['usd_high'] for r in rows),
            'input_low': sum(r['input_low'] for r in rows), 'input_high': sum(r['input_high'] for r in rows),
            'output_max': sum(r['output_max'] for r in rows)}


def describe(report):
    if not report['available']:
        return [f"Token/dollar estimate unavailable: {report['reason']}"]
    lines = ['Whole-plan token estimate (before cache savings; not remaining resume cost):']
    for row in report['groups']:
        mode = 'batch' if row['batch'] else 'standard'
        lines.append(f"  {row['role']}: {row['model']} ({mode}, {row['encoding']}), "
                     f"{row['input_low']:,}–{row['input_high']:,} input tokens; "
                     f"up to {row['output_max']:,} output tokens")
    lines += [f"Estimated whole-plan cost: USD ${report['usd_low']:.2f}–${report['usd_high']:.2f} "
              f"(rates verified {report['price_date']}).",
              'Range: known inputs with no output, through candidate allowance + maximum output (including reasoning).',
              'Counts include prompt text, schema and estimated framing; candidate wording is not yet known unless saved.',
              'Not a billing cap. No cache discounts; excludes retries, tax, regional surcharges and cache-write premiums.',
              f'Pricing: {PRICE_SOURCE}; batch: {BATCH_SOURCE}']
    return lines
