from pathlib import Path
import pandas as pd

class GenericDirectoryLogAdapter:
    """Reads .log/.txt/.out files recursively and returns a common dataframe.

    Output columns: session_id, timestamp, raw_message, label, service, source_file.
    Label is inferred from path words: error/fault/anomaly/abnormal/failure/fail -> 1.
    For real LO2/RCAEval layouts, refine _infer_label_from_path/session_id logic here.
    """
    exts = {'.log', '.txt', '.out'}
    anomaly_words = ['error', 'fault', 'anomaly', 'abnormal', 'failure', 'fail']

    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)

    def _infer_label_from_path(self, path: Path):
        s = str(path).lower()
        return int(any(w in s for w in self.anomaly_words))

    def load(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f'Raw directory not found: {self.raw_dir}')
        files = [p for p in self.raw_dir.rglob('*') if p.suffix.lower() in self.exts]
        if not files:
            raise FileNotFoundError(f'No .log/.txt/.out files under {self.raw_dir}')
        rows = []
        for fp in files:
            label = self._infer_label_from_path(fp)
            session_id = fp.stem
            service = fp.parent.name
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    msg = line.strip()
                    if msg:
                        rows.append({'session_id': session_id, 'timestamp': i, 'raw_message': msg,
                                     'label': label, 'service': service, 'source_file': str(fp)})
        return pd.DataFrame(rows)

class LO2Adapter(GenericDirectoryLogAdapter):
    pass

class LoghubAdapter(GenericDirectoryLogAdapter):
    pass

class RCAEvalAdapter(GenericDirectoryLogAdapter):
    pass

def get_adapter(dataset_name, raw_dir):
    name = dataset_name.lower()
    if name in ['demo', 'generic']:
        return GenericDirectoryLogAdapter(raw_dir)
    if name == 'lo2':
        return LO2Adapter(raw_dir)
    if name in ['loghub', 'loghub2', 'loghub-2.0']:
        return LoghubAdapter(raw_dir)
    if name == 'rcaeval':
        return RCAEvalAdapter(raw_dir)
    raise ValueError(f'Unknown dataset_name: {dataset_name}')
