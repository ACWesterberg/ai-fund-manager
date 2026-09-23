"""Durable OpenAI evaluation batches. Never silently resubmit an uncertain batch."""
import json
from pathlib import Path
from types import SimpleNamespace

from fundmgr.engine.client import LLMError, _report_usage, _schema_hint
from fundmgr.engine.experiments import digest
from fundmgr.engine.schema import DecisionRun


class BatchPending(RuntimeError):
    """Normal asynchronous completion: caller should resume later."""


def batch_client():
    from openai import OpenAI
    return OpenAI(max_retries=0)


def body_for(descriptor):
    from openai.lib._pydantic import to_strict_json_schema
    llm = descriptor['llm']
    new_api = llm['model_id'].startswith(('gpt-5', 'o1', 'o3', 'o4'))
    body = {'model': llm['model_id'], 'messages': [
        {'role': 'system', 'content': descriptor['system'] + _schema_hint(DecisionRun)},
        {'role': 'user', 'content': descriptor['user']}],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'DecisionRun', 'strict': True, 'schema': to_strict_json_schema(DecisionRun)}},
        'max_completion_tokens' if new_api else 'max_tokens': llm['max_tokens']}
    if not new_api:
        body['temperature'] = llm['temperature']
    if new_api and llm.get('reasoning_effort'):
        body['reasoning_effort'] = llm['reasoning_effort']
    return body


def process_batch(cfg, plan, path, state, pending, *, retry_failed=False, adopt_id=None, collect_only=False):
    from fundmgr.engine.bounded_optimizer import _save, SearchStopped
    batches = state.setdefault('batches', [])
    batch = batches[-1] if batches and not batches[-1].get('collected') else None
    if batch is None:
        if adopt_id:
            raise ValueError('--batch-id requires an unresolved saved submission')
        if not pending:
            return
        if collect_only:
            raise SearchStopped("Collection cannot submit or retry batch requests; manual resume required")
        grouped = {}
        for job in pending:
            group_key = job['cache_key'] if cfg.optimizer.reuse_evaluations else job['key']
            existing = grouped.get(group_key)
            if existing:
                existing['keys'].append(job['key'])
            else:
                grouped[group_key] = {**job, 'keys': [job['key']]}
        jobs = list(grouped.values())
        for job in jobs:
            if any(a['key'] in job['keys'] for a in state['attempts']) and not retry_failed:
                raise SearchStopped('Failed batch requests require --retry-failed; successful results remain saved')
        amount = sum(job['reserved_tokens'] for job in jobs)
        if (len(state['attempts']) + len(jobs) > cfg.optimizer.max_calls or
                sum(a['reserved_tokens'] for a in state['attempts']) + amount > cfg.optimizer.max_total_tokens):
            raise SearchStopped('Budget exhausted; batch not submitted. Limits are cumulative across resumes')
        # Validate all request bodies locally before creating any paid request.
        lines = []
        for i, job in enumerate(jobs):
            job['custom_id'] = f'eval-{i}'
            lines.append({'custom_id': job['custom_id'], 'method': 'POST',
                          'url': '/v1/chat/completions', 'body': body_for(job['descriptor'])})
        client = batch_client()
        payload = ('\n'.join(json.dumps(line, ensure_ascii=False) for line in lines) + '\n').encode()
        uploaded = client.files.create(file=('optimizer.jsonl', payload, 'application/jsonl'), purpose='batch')
        batch = {'status': 'submitting', 'file_id': uploaded.id, 'jobs': jobs,
                 'metadata': {'optimizer_run': digest(plan), 'group': str(len(batches))}}
        for job in jobs:
            job['attempt_index'] = len(state['attempts'])
            state['attempts'].append({'key': job['key'], 'model': job['descriptor']['llm']['model_id'],
                                     'reserved_tokens': job['reserved_tokens'], 'status': 'in_flight',
                                     'usage': None, 'transport': 'openai_batch'})
        batches.append(batch)
        _save(path, state)  # A crash after this point must never cause automatic resubmission.
        try:
            submitted = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/chat/completions',
                                              completion_window='24h', metadata=batch['metadata'])
        except Exception as exc:
            raise SearchStopped('Batch submission uncertain. Recover its ID from the provider dashboard and '
                                'resume with --batch-id; no automatic resubmission.') from exc
        batch.update(id=submitted.id, status=submitted.status)
        state['status'] = 'batch_pending'
        _save(path, state)
        raise BatchPending(f'Batch {submitted.id} submitted. Resume later with --resume {path}')

    client = batch_client()
    if adopt_id and batch.get('id') and adopt_id != batch['id']:
        raise ValueError('Batch ID does not match saved submission')
    batch_id = batch.get('id') or adopt_id
    if not batch_id:
        raise SearchStopped('Uncertain batch submission: resume with --batch-id from the provider dashboard; '
                            '--retry-failed does not resubmit it')
    remote = client.batches.retrieve(batch_id)
    if remote.input_file_id != batch['file_id'] or remote.metadata != batch['metadata']:
        raise ValueError('Remote batch does not match checkpoint input file and metadata')
    batch.update(id=batch_id, status=remote.status)
    _save(path, state)
    if remote.status not in ('completed', 'failed', 'expired', 'cancelled'):
        state['status'] = 'batch_pending'
        _save(path, state)
        raise BatchPending(f'Batch {batch_id}: {remote.status}. Resume later with --resume {path}')

    # Read both files; output order is unspecified. Validate IDs before attaching results.
    records = {}
    expected = {job['custom_id'] for job in batch['jobs']}
    for file_id in (remote.output_file_id, remote.error_file_id):
        if not file_id:
            continue
        for line in client.files.content(file_id).text.splitlines():
            record = json.loads(line)
            key = record['custom_id']
            if key not in expected or key in records:
                raise ValueError('Unexpected or duplicate batch result ID')
            records[key] = record
    errors = []
    for job in batch['jobs']:
        attempt = state['attempts'][job['attempt_index']]
        if attempt['status'] == 'complete':
            continue  # Crash during collection: already saved, no double accounting.
        record = records.get(job['custom_id'], {})
        response = record.get('response') or {}
        body = response.get('body') or {}
        def usage_callback(usage):
            attempt['usage'] = usage
            _save(path, state)
        usage = body.get('usage')
        _report_usage(SimpleNamespace(usage=SimpleNamespace(**usage) if usage else None,
                                     id=body.get('id'), model=body.get('model')), 'openai', usage_callback)
        try:
            if record.get('error') or response.get('status_code') != 200:
                raise LLMError(f'Batch request failed: {record.get("error") or response.get("status_code") or remote.status}')
            choice = body['choices'][0]
            if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
                raise LLMError('Batch completion refused or truncated')
            raw = choice['message']['content']
            parsed = DecisionRun.model_validate_json(raw).model_dump()
        except (ValueError, KeyError, IndexError, TypeError, LLMError) as exc:
            attempt.update(status='failed', error=str(exc)[:500])
            errors.append(job['key'])
            _save(path, state)
            continue
        result = {'parsed': parsed, 'raw': raw, 'origin': 'provider', 'cache_key': job['cache_key']}
        for key in job['keys']:
            state['results'][key] = result
        attempt['status'] = 'complete'
        _save(path, state)
        if cfg.optimizer.reuse_evaluations:
            cache = Path(job['cache_path'])
            cache.parent.mkdir(parents=True, exist_ok=True)
            _save(cache, {'request': job['descriptor'], 'parsed': parsed, 'raw': raw,
                          'source_checkpoint': str(path), 'source_attempt': job['attempt_index']})
    batch['collected'] = True
    _save(path, state)
    if errors:
        raise SearchStopped(f'{len(errors)} batch request(s) failed; successes saved. '
                            'Use --retry-failed with sufficient cumulative budget to retry only failures')
