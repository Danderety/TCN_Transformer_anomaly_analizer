import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import random
from pathlib import Path

def make_line(event, request_id, step, error=False):
    level = 'ERROR' if error and event in ['DB_QUERY', 'PAYMENT', 'DB_CORRUPTION', 'UNKNOWN_ADMIN_ACTION'] else 'INFO'
    status = 'failed code=500' if level == 'ERROR' else 'ok'
    return f'2026-01-01 10:00:{step:02d} {level} request={request_id} event={event} {status} latency={random.randint(5,900)}ms'

def main(output_dir='data/raw/demo', n_normal=600, n_anomaly=180):
    random.seed(42)
    out = Path(output_dir)
    (out / 'normal').mkdir(parents=True, exist_ok=True)
    (out / 'error').mkdir(parents=True, exist_ok=True)
    patterns = [
        ['AUTH_START', 'TOKEN_CHECK', 'USER_LOOKUP', 'PERMISSION_CHECK', 'DB_QUERY', 'RESPONSE_SEND'],
        ['AUTH_START', 'TOKEN_CHECK', 'USER_LOOKUP', 'CACHE_HIT', 'RESPONSE_SEND'],
        ['AUTH_START', 'TOKEN_CHECK', 'USER_LOOKUP', 'PAYMENT', 'DB_QUERY', 'RESPONSE_SEND'],
    ]
    for i in range(n_normal):
        pat = random.choice(patterns)
        lines = [make_line(e, f'normal_{i}', j) for j, e in enumerate(pat)]
        (out / 'normal' / f'session_{i:05d}.log').write_text('\n'.join(lines), encoding='utf-8')
    for i in range(n_anomaly):
        pat = random.choice(patterns).copy()
        t = random.choice(['insert', 'swap', 'replace', 'repeat'])
        if t == 'insert':
            pat.insert(random.randint(1, len(pat) - 1), 'UNKNOWN_ADMIN_ACTION')
        elif t == 'swap':
            pos = random.randint(1, len(pat) - 3)
            pat[pos], pat[pos + 1] = pat[pos + 1], pat[pos]
        elif t == 'replace':
            pat[random.randint(1, len(pat) - 2)] = 'DB_CORRUPTION'
        elif t == 'repeat':
            pos = random.randint(1, len(pat) - 2)
            pat.insert(pos, pat[pos])
        lines = [make_line(e, f'error_{i}', j, True) for j, e in enumerate(pat)]
        (out / 'error' / f'session_{i:05d}.log').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Demo data generated: {out}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='data/raw/demo')
    p.add_argument('--n_normal', type=int, default=600)
    p.add_argument('--n_anomaly', type=int, default=180)
    a = p.parse_args()
    main(a.output_dir, a.n_normal, a.n_anomaly)
