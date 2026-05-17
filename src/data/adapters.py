from pathlib import Path
import re
import pandas as pd

class GenericDirectoryLogAdapter:
    """Reads .log/.txt/.out files recursively and returns a common dataframe.

    Output columns: session_id, timestamp, raw_message, label, service, source_file.
    Label is inferred from path words: error/fault/anomaly/abnormal/failure/fail -> 1.
    For real LO2/RCAEval layouts, refine _infer_label_from_path/session_id logic here.
    """
    exts = {'.log', '.txt', '.out'}
    anomaly_words = ['error', 'fault', 'anomaly', 'abnormal', 'failure', 'fail']

    def __init__(self, raw_dir, max_files=None, max_lines_per_file=None, **kwargs):
        self.raw_dir = Path(raw_dir)
        self.max_files = max_files
        self.max_lines_per_file = max_lines_per_file

    def _infer_label_from_path(self, path: Path):
        s = str(path).lower()
        return int(any(w in s for w in self.anomaly_words))

    def load(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f'Raw directory not found: {self.raw_dir}')
        files = [p for p in self.raw_dir.rglob('*') if p.suffix.lower() in self.exts]
        if self.max_files:
            files = files[:int(self.max_files)]
        if not files:
            raise FileNotFoundError(f'No .log/.txt/.out files under {self.raw_dir}')
        rows = []
        for fp in files:
            label = self._infer_label_from_path(fp)
            session_id = fp.stem
            service = fp.parent.name
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if self.max_lines_per_file and i >= int(self.max_lines_per_file):
                        break
                    msg = line.strip()
                    if msg:
                        rows.append({'session_id': session_id, 'timestamp': i, 'raw_message': msg,
                                     'label': label, 'service': service, 'source_file': str(fp)})
        return pd.DataFrame(rows)

class LO2Adapter(GenericDirectoryLogAdapter):
    def __init__(self, raw_dir, max_files=None, max_lines_per_file=None, max_runs_per_scenario=None, session_granularity='scenario', **kwargs):
        super().__init__(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
        self.max_runs_per_scenario = max_runs_per_scenario
        self.session_granularity = str(session_granularity or 'scenario').lower()

    def _parts(self, path):
        rel = path.relative_to(self.raw_dir)
        parts = rel.parts
        if parts and parts[0] == 'logs' and len(parts) >= 4:
            return parts[1], parts[2], path.stem
        if len(parts) >= 3:
            return parts[0], parts[1], path.stem
        return 'unknown_run', path.parent.name, path.stem

    def _infer_label_from_path(self, path):
        _run, scenario, _service = self._parts(path)
        return int(scenario.lower() != 'correct')

    def load(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f'Raw directory not found: {self.raw_dir}')
        files = [p for p in self.raw_dir.rglob('*.log')]
        files = sorted(files, key=lambda p: self._parts(p))
        if self.max_runs_per_scenario:
            allowed = {}
            for fp in files:
                run_id, scenario, _service = self._parts(fp)
                runs = allowed.setdefault(scenario, [])
                if run_id not in runs and len(runs) < int(self.max_runs_per_scenario):
                    runs.append(run_id)
            files = [fp for fp in files if self._parts(fp)[0] in set(allowed.get(self._parts(fp)[1], []))]
        if self.max_files:
            files = files[:int(self.max_files)]
        if not files:
            raise FileNotFoundError(f'No .log files under {self.raw_dir}')
        rows = []
        service_order = {}
        for fp in files:
            run_id, scenario, service = self._parts(fp)
            label = self._infer_label_from_path(fp)
            if self.session_granularity == 'service':
                session_id = f'{run_id}/{scenario}/{service}'
            else:
                session_id = f'{run_id}/{scenario}'
            service_rank = service_order.setdefault(service, len(service_order))
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if self.max_lines_per_file and i >= int(self.max_lines_per_file):
                        break
                    msg = line.strip()
                    if msg:
                        rows.append({
                            'session_id': session_id,
                            'timestamp': i * 100 + service_rank,
                            'raw_message': msg,
                            'label': label,
                            'service': service,
                            'source_file': str(fp),
                            'run_id': run_id,
                            'scenario': scenario,
                        })
        return pd.DataFrame(rows)

class LoghubAdapter(GenericDirectoryLogAdapter):
    block_pattern = re.compile(r'blk_-?\d+')

    def _find_label_path(self):
        candidates = []
        for name in ['anomaly_label.csv', 'HDFS_anomaly_label.csv']:
            candidates.append(self.raw_dir / name)
            candidates.append(self.raw_dir / 'preprocessed' / name)
        candidates.extend(self.raw_dir.rglob('anomaly_label.csv'))
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _load_block_labels(self, label_path):
        labels = pd.read_csv(label_path)
        if 'BlockId' in labels.columns:
            block_col = 'BlockId'
            label_cols = [c for c in labels.columns if c != block_col]
            label_col = 'Label' if 'Label' in label_cols else label_cols[0]
            raw = labels.set_index(block_col)[label_col]
        else:
            labels = pd.read_csv(label_path, index_col=0)
            raw = labels.iloc[:, 0]
        normalized = raw.astype(str).str.strip().str.lower()
        mapped = normalized.map({'normal': 0, 'anomaly': 1, '0': 0, '1': 1})
        if mapped.isna().any():
            bad = sorted(normalized[mapped.isna()].unique().tolist())
            raise ValueError(f'Unknown HDFS labels in {label_path}: {bad}')
        return {str(k): int(v) for k, v in mapped.items()}

    def _extract_block_id(self, text):
        match = self.block_pattern.search(str(text))
        return match.group(0) if match else None

    def load(self):
        structured = self.raw_dir / 'HDFS_full.log_structured.csv'
        if structured.exists():
            read_kwargs = {'usecols': ['LineId', 'Content', 'EventId']}
            if self.max_lines_per_file:
                read_kwargs['nrows'] = int(self.max_lines_per_file)
            df = pd.read_csv(structured, **read_kwargs)
            df = df.rename(columns={'LineId': 'timestamp', 'Content': 'raw_message'})
            label_path = self._find_label_path()
            if label_path is not None:
                labels = self._load_block_labels(label_path)
                df['session_id'] = df['raw_message'].map(self._extract_block_id)
                df = df[df['session_id'].isin(labels)].copy()
                if df.empty:
                    raise ValueError(f'No HDFS structured rows matched labels from {label_path}')
                df['label'] = df['session_id'].map(labels).astype(int)
            else:
                # No gold label file is present, so keep the stream as unlabeled
                # normal data for unsupervised training.
                df['session_id'] = (df['timestamp'].astype(int) // 128).map(lambda x: f'hdfs_stream_{x}')
                df['label'] = 0
            df['service'] = 'hdfs'
            df['source_file'] = str(structured)
            return df[['session_id', 'timestamp', 'raw_message', 'label', 'service', 'source_file']]
        return super().load()

class RCAEvalAdapter(GenericDirectoryLogAdapter):
    def load(self):
        files = [p for p in self.raw_dir.rglob('data.csv')]
        if self.max_files:
            files = files[:int(self.max_files)]
        if not files:
            return super().load()
        rows = []
        for fp in files:
            df = pd.read_csv(fp)
            inject_path = fp.parent / 'inject_time.txt'
            inject_time = None
            if inject_path.exists():
                text = inject_path.read_text(encoding='utf-8', errors='ignore').strip()
                try:
                    inject_time = float(text.split()[0])
                except (IndexError, ValueError):
                    inject_time = None
            time_col = 'time'
            if time_col not in df.columns:
                df.insert(0, time_col, range(len(df)))
            numeric = df.select_dtypes(include='number').copy()
            numeric = numeric.loc[:, ~numeric.columns.duplicated()]
            metric_cols = [c for c in numeric.columns if c != time_col]
            if self.max_lines_per_file:
                df = df.head(int(self.max_lines_per_file))
            service = fp.parent.parent.name
            run_id = fp.parent.name
            for i, row in df.iterrows():
                ts = float(row[time_col]) if time_col in row else float(i)
                label = int(inject_time is not None and ts >= inject_time)
                window = int(i) // 128
                session_id = f'{service}/{run_id}/window_{window:04d}'
                # Convert metrics into compact pseudo-log events. This keeps the
                # downstream log-token pipeline unchanged while preserving signal.
                values = pd.to_numeric(row[metric_cols], errors='coerce').dropna()
                if values.empty:
                    messages = ['metrics heartbeat']
                else:
                    top = values.abs().sort_values(ascending=False).head(5)
                    messages = [f'metric {name} value_bin {int(value // 1) if abs(value) < 1000 else "large"}' for name, value in top.items()]
                for j, msg in enumerate(messages):
                    rows.append({
                        'session_id': session_id,
                        'timestamp': int(i) * 10 + j,
                        'raw_message': msg,
                        'label': label,
                        'service': service,
                        'source_file': str(fp),
                    })
        return pd.DataFrame(rows)

def get_adapter(dataset_name, raw_dir, max_files=None, max_lines_per_file=None, max_runs_per_scenario=None, session_granularity=None):
    name = dataset_name.lower()
    if name in ['demo', 'generic']:
        return GenericDirectoryLogAdapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
    if name == 'lo2':
        return LO2Adapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file, max_runs_per_scenario=max_runs_per_scenario, session_granularity=session_granularity or 'scenario')
    if name in ['loghub', 'loghub2', 'loghub-2.0']:
        return LoghubAdapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
    if name == 'rcaeval':
        return RCAEvalAdapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
    raise ValueError(f'Unknown dataset_name: {dataset_name}')
