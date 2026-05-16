from pathlib import Path

from src.utils.io import load_json


def has_anomalies(payload):
    return any(int(y) == 1 for y in payload.get('labels', []))


def combine_validation_with_synthetic_if_needed(config, val_payload):
    """Use synthetic validation anomalies when real validation has no positives."""
    enabled = bool(config.get('evaluation', {}).get('use_synthetic_validation_when_no_anomalies', True))
    if not enabled or has_anomalies(val_payload):
        return val_payload, 'real_val'

    splits_dir = Path(config['data']['splits_dir'])
    synth_path = splits_dir / 'val_synthetic.json'
    if not synth_path.exists():
        return val_payload, 'real_val_no_synthetic_available'

    synth = load_json(synth_path)
    combined = {
        'sequences': val_payload.get('sequences', []) + synth.get('sequences', []),
        'labels': val_payload.get('labels', []) + synth.get('labels', []),
        'session_ids': val_payload.get('session_ids', []) + synth.get('session_ids', []),
    }
    return combined, 'real_val_plus_val_synthetic'
