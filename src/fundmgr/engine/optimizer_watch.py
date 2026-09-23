"""Poll existing fund batches and notify on completion or required intervention."""
import fcntl
import logging

from fundmgr.engine.bounded_optimizer import _load, _save, run_search, SearchBusy
from fundmgr.engine.optimizer_batch import BatchPending
from fundmgr.engine.client import LLMError
from fundmgr.engine.experiments import digest
from fundmgr.notify.send import send_telegram

logger = logging.getLogger(__name__)


def watch(cfg):
    root = cfg.optimizer.compiled_dir / 'searches' / cfg.db_path.stem
    counts = dict(checked=0, pending=0, complete=0, attention=0, notified=0, notification_failed=0)
    if not root.exists():
        return counts
    # Separate lock/receipt files never alter search accounting or its checksum.
    receipts = root / 'notifications'
    receipts.mkdir(exist_ok=True)
    with (root / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return counts
        for path in sorted(root.glob('*.json')):
            try:
                state = _load(path)
                if state['plan'].get('execution') != 'batch' or not state.get('batches'):
                    continue
                counts['checked'] += 1
                if state['status'] != 'complete':
                    run_search(cfg, state['plan'], path, resume=True, collect_only=True)
                    state = _load(path)
                if state['status'] != 'complete':
                    raise RuntimeError('Collection did not finish')
                counts['complete'] += 1
                if state['plan'].get('context_mode') == 'compare':
                    outcome = 'Context comparison finished; review paired decisions.'
                elif state.get('candidate_path'):
                    outcome = 'Improved prompt candidate saved; forward evaluation is still required.'
                else:
                    outcome = 'Search finished; the incumbent prompt was retained.'
                kind = 'complete'
                event = digest([kind, state.get('candidate_path'), state.get('comparisons')])
            except (BatchPending, SearchBusy):
                counts['pending'] += 1
                continue  # Stay quiet while pending or another process owns the fund.
            except LLMError as exc:
                # Transport failures are retried by the next check, never by a new
                # model request. Avoid alerting on each transient network outage.
                logger.warning('Batch check unavailable for %s: %s', path.name, exc)
                counts['pending'] += 1
                continue
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                counts['attention'] += 1
                logger.warning('Batch collection needs attention for %s: %s', path.name, exc)
                kind = 'attention'
                event = digest([kind, type(exc).__name__])
                outcome = ('Batch collection needs attention. Inspect the optimizer watch log and resume manually. '
                           'No failed requests were retried.')
            receipt = receipts / path.name
            previous = _load(receipt) if receipt.exists() else {}
            if previous.get('event') == event:
                continue
            message = (f'Prompt optimizer — {cfg.db_path.stem}\n{outcome}\n'
                       f'No guidance was activated by this check.\nCheckpoint: {path}')
            if send_telegram(message, parse_mode=''):
                _save(receipt, {'event': event, 'kind': kind})
                counts['notified'] += 1
            else:
                # Retry delivery next check, without repeating paid work.
                counts['notification_failed'] += 1
                logger.warning('Telegram delivery failed for %s; will retry next check', path.name)
    return counts
