"""
Governance / system-contract helpers.

Compatibility-first implementation of the v2 system context:
  - pipeline signatures
  - feature/version metadata
  - model registry append-only records
  - lightweight audit trail

These helpers are intentionally additive so existing notebooks can keep calling
the same `save_model()` / `load_model()` / signal functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats with readable strings."""
    if isinstance(value, float):
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    return value


def get_feature_contract() -> dict[str, Any]:
    """
    Lightweight feature contract until the repo adopts a full feature registry.
    """
    return {
        'feature_version': config.FEATURE_VERSION,
        'feature_groups': ['macro', 'anchor', 'altcoin'],
        'feature_pipeline': 'utils.features',
    }


def get_label_contract() -> dict[str, Any]:
    return {
        'label_logic_version': config.LABEL_LOGIC_VERSION,
        'horizon': config.LABEL_HORIZON,
        'tp_grid': config.TP_GRID,
        'sl_grid': config.SL_GRID,
        'k1_grid': config.K1_GRID,
        'k2_grid': config.K2_GRID,
    }


def generate_pipeline_signature(
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        'pipeline_version': config.PIPELINE_VERSION,
        'data_schema_version': config.DATA_SCHEMA_VERSION,
        'timeframe': config.TIMEFRAME,
        'thresholds': {
            'long': config.LONG_THRESHOLD,
            'short': config.SHORT_THRESHOLD,
        },
        'feature_contract': get_feature_contract(),
        'label_contract': get_label_contract(),
    }
    if extra:
        payload['extra'] = extra
    return hashlib.sha256(_stable_json(payload).encode('utf-8')).hexdigest()


def validate_pipeline_signature(model_signature: str | None) -> tuple[bool, str]:
    runtime_signature = generate_pipeline_signature()
    if not model_signature:
        if config.ENFORCE_PIPELINE_SIGNATURE:
            return False, 'missing pipeline_signature'
        return True, 'pipeline_signature missing but enforcement disabled'
    if model_signature != runtime_signature:
        if config.ENFORCE_PIPELINE_SIGNATURE:
            return False, 'pipeline_signature mismatch'
        return True, 'pipeline_signature mismatch but enforcement disabled'
    return True, 'ok'


def _registry_path() -> str:
    return os.path.join(config.MODELS_DIR, 'model_registry.jsonl')


def _audit_path() -> str:
    return os.path.join(config.LOGS_DIR, 'audit_log.jsonl')


def append_model_registry(record: dict[str, Any]) -> None:
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(_registry_path(), 'a', encoding='utf-8') as fh:
        fh.write(_stable_json(sanitize_for_json(record)) + '\n')


def audit_event(event_type: str, payload: dict[str, Any]) -> None:
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    record = {
        'timestamp': _now_utc_iso(),
        'event_type': event_type,
        'payload': sanitize_for_json(payload),
    }
    with open(_audit_path(), 'a', encoding='utf-8') as fh:
        fh.write(_stable_json(record) + '\n')


def build_model_metadata(ticker: str, meta: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade caller-supplied metadata into the compatibility v2 schema.
    """
    safe_meta = sanitize_for_json(meta)
    out = dict(safe_meta)
    out.setdefault('ticker', ticker)
    out.setdefault('model_meta_version', config.MODEL_META_VERSION)
    out.setdefault('pipeline_version', config.PIPELINE_VERSION)
    out.setdefault('data_schema_version', config.DATA_SCHEMA_VERSION)
    out.setdefault('pipeline_signature', generate_pipeline_signature())
    out.setdefault('feature_versions', get_feature_contract())
    out.setdefault('label_config', out.get('label_params', {}))
    out.setdefault('training_config', {
        'final_estimators': config.FINAL_ESTIMATORS,
        'grid_search_estimators': config.GRID_SEARCH_ESTIMATORS,
        'min_trade_count': config.MIN_TRADE_COUNT,
        'min_pf_test1': config.MIN_PF_TEST1,
        'pf_stability_ratio': config.PF_STABILITY_RATIO,
        'max_drawdown': config.MAX_DRAWDOWN,
    })
    out.setdefault('trained_at', _now_utc_iso())
    out.setdefault('model_id', f"{ticker}|{out['trained_at']}|{out['pipeline_signature'][:12]}")
    return out
