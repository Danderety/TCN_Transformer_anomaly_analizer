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

    def __init__(
        self,
        raw_dir,
        max_files=None,
        max_lines_per_file=None,
        use_official_event_id=False,
        block_features=False,
        max_blocks=None,
        max_normal_blocks=None,
        max_anomaly_blocks=None,
        chunk_size=500000,
        rare_transition_max_count=2,
        transition_top_k=32,
        **kwargs,
    ):
        super().__init__(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
        self.use_official_event_id = bool(use_official_event_id)
        self.block_features = bool(block_features)
        self.max_blocks = max_blocks
        self.max_normal_blocks = max_normal_blocks
        self.max_anomaly_blocks = max_anomaly_blocks
        self.chunk_size = int(chunk_size)
        self.rare_transition_max_count = int(rare_transition_max_count)
        self.transition_top_k = int(transition_top_k)

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

    def _select_blocks(self, labels):
        if not (self.max_blocks or self.max_normal_blocks or self.max_anomaly_blocks):
            return None
        normal = [block for block, label in labels.items() if int(label) == 0]
        anomaly = [block for block, label in labels.items() if int(label) == 1]
        normal = sorted(normal)
        anomaly = sorted(anomaly)
        if self.max_normal_blocks is not None:
            normal = normal[:int(self.max_normal_blocks)]
        if self.max_anomaly_blocks is not None:
            anomaly = anomaly[:int(self.max_anomaly_blocks)]
        selected = normal + anomaly
        if self.max_blocks is not None and len(selected) > int(self.max_blocks):
            selected = selected[:int(self.max_blocks)]
        return set(selected)

    def _event_token(self, event_id):
        event_id = str(event_id).strip() or 'UNKNOWN'
        digit_words = {
            '0': 'ZERO',
            '1': 'ONE',
            '2': 'TWO',
            '3': 'THREE',
            '4': 'FOUR',
            '5': 'FIVE',
            '6': 'SIX',
            '7': 'SEVEN',
            '8': 'EIGHT',
            '9': 'NINE',
        }
        safe = ''.join(digit_words.get(ch, ch) for ch in event_id)
        safe = re.sub(r'[^A-Za-z_]+', '_', safe).strip('_') or 'UNKNOWN'
        return f'event_{safe}'

    def _count_bin(self, value):
        value = int(value)
        if value <= 1:
            return 'once'
        if value <= 3:
            return 'few'
        if value <= 8:
            return 'some'
        if value <= 20:
            return 'many'
        return 'burst'

    def _length_bin(self, value):
        value = int(value)
        if value <= 5:
            return 'tiny'
        if value <= 20:
            return 'short'
        if value <= 80:
            return 'medium'
        if value <= 200:
            return 'long'
        return 'huge'

    def _add_block_features(self, df):
        if df.empty or 'event_id_raw' not in df.columns:
            return df
        rows = []
        normal_pair_counts = {}
        for _sid, g in df[df['label'] == 0].sort_values(['session_id', 'timestamp']).groupby('session_id', sort=False):
            events = [str(x) for x in g['event_id_raw'].tolist()]
            for a, b in zip(events, events[1:]):
                normal_pair_counts[(a, b)] = normal_pair_counts.get((a, b), 0) + 1
        for sid, g in df.sort_values(['session_id', 'timestamp']).groupby('session_id', sort=False):
            events = [str(x) for x in g['event_id_raw'].tolist()]
            if not events:
                continue
            label = int(g['label'].max())
            service = str(g['service'].iloc[0])
            source_file = str(g['source_file'].iloc[0])
            base_ts = int(float(g['timestamp'].max())) + 1
            messages = [
                f'block_length_{self._length_bin(len(events))}',
                f'block_first_{self._event_token(events[0])}',
                f'block_last_{self._event_token(events[-1])}',
            ]
            counts = pd.Series(events).value_counts()
            for event_id, count in counts.items():
                messages.append(f'block_count_{self._event_token(event_id)}_{self._count_bin(count)}')
            pair_counts = {}
            for a, b in zip(events, events[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            top_pairs = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:self.transition_top_k]
            for (a, b), count in top_pairs:
                a_token = self._event_token(a)
                b_token = self._event_token(b)
                messages.append(f'transition_{a_token}_TO_{b_token}_{self._count_bin(count)}')
                if normal_pair_counts.get((a, b), 0) <= self.rare_transition_max_count:
                    messages.append(f'rare_transition_{a_token}_TO_{b_token}')
            for offset, msg in enumerate(messages):
                rows.append({
                    'session_id': sid,
                    'timestamp': base_ts + offset,
                    'raw_message': msg,
                    'label': label,
                    'service': service,
                    'source_file': source_file,
                    'event_id_raw': msg,
                    'is_block_feature': True,
                })
        if not rows:
            return df
        base = df.copy()
        base['is_block_feature'] = False
        return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)

    def load(self):
        structured = self.raw_dir / 'HDFS_full.log_structured.csv'
        if structured.exists():
            label_path = self._find_label_path()
            if label_path is not None:
                labels = self._load_block_labels(label_path)
                selected_blocks = self._select_blocks(labels)
                if selected_blocks is not None or not self.max_lines_per_file:
                    parts = []
                    for chunk in pd.read_csv(structured, usecols=['LineId', 'Content', 'EventId'], chunksize=self.chunk_size):
                        chunk = chunk.rename(columns={'LineId': 'timestamp', 'Content': 'raw_message'})
                        chunk['session_id'] = chunk['raw_message'].map(self._extract_block_id)
                        keep = chunk['session_id'].isin(labels)
                        if selected_blocks is not None:
                            keep &= chunk['session_id'].isin(selected_blocks)
                        chunk = chunk[keep].copy()
                        if not chunk.empty:
                            parts.append(chunk)
                    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['timestamp', 'raw_message', 'EventId', 'session_id'])
                else:
                    read_kwargs = {'usecols': ['LineId', 'Content', 'EventId'], 'nrows': int(self.max_lines_per_file)}
                    df = pd.read_csv(structured, **read_kwargs)
                    df = df.rename(columns={'LineId': 'timestamp', 'Content': 'raw_message'})
                    df['session_id'] = df['raw_message'].map(self._extract_block_id)
                    df = df[df['session_id'].isin(labels)].copy()
                if df.empty:
                    raise ValueError(f'No HDFS structured rows matched labels from {label_path}')
                df['label'] = df['session_id'].map(labels).astype(int)
                df['event_id_raw'] = df['EventId'].astype(str)
                if self.use_official_event_id:
                    df['raw_message'] = df['event_id_raw'].map(self._event_token)
                df['service'] = 'hdfs'
                df['source_file'] = str(structured)
                if self.block_features:
                    df = self._add_block_features(df)
            else:
                read_kwargs = {'usecols': ['LineId', 'Content', 'EventId']}
                if self.max_lines_per_file:
                    read_kwargs['nrows'] = int(self.max_lines_per_file)
                df = pd.read_csv(structured, **read_kwargs)
                df = df.rename(columns={'LineId': 'timestamp', 'Content': 'raw_message'})
                # No gold label file is present, so keep the stream as unlabeled
                # normal data for unsupervised training.
                df['session_id'] = (df['timestamp'].astype(int) // 128).map(lambda x: f'hdfs_stream_{x}')
                df['label'] = 0
            df['service'] = 'hdfs'
            df['source_file'] = str(structured)
            return df[['session_id', 'timestamp', 'raw_message', 'label', 'service', 'source_file']]
        return super().load()

class RCAEvalAdapter(GenericDirectoryLogAdapter):
    def __init__(
        self,
        raw_dir,
        max_files=None,
        max_lines_per_file=None,
        window_size=128,
        top_k=8,
        abnormal_z=2.0,
        sparse_events=False,
        row_features=True,
        window_features=False,
        max_metric_tokens_per_row=16,
        **kwargs,
    ):
        super().__init__(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
        self.window_size = int(window_size or 128)
        self.top_k = int(top_k or 8)
        self.abnormal_z = float(abnormal_z)
        self.sparse_events = bool(sparse_events)
        self.row_features = bool(row_features)
        self.window_features = bool(window_features)
        self.max_metric_tokens_per_row = int(max_metric_tokens_per_row or top_k or 8)

    def _safe_token(self, value, prefix='metric'):
        text = str(value).strip() or 'unknown'
        digit_words = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine',
        }
        text = ''.join(digit_words.get(ch, ch) for ch in text)
        text = re.sub(r'[^A-Za-z_]+', '_', text).strip('_').lower() or 'unknown'
        return f'{prefix}_{text}'

    def _metric_type(self, name):
        parts = [p for p in re.split(r'[_\W]+', str(name).lower()) if p]
        if not parts:
            return 'metric'
        for key in ['cpu', 'mem', 'memory', 'load', 'latency', 'delay', 'error', 'disk', 'net', 'network', 'qps', 'rate']:
            if key in parts:
                return key
        return parts[-1]

    def _metric_level(self, z):
        z = abs(float(z))
        if z >= 8:
            return 'extreme'
        if z >= 4:
            return 'high'
        if z >= 2:
            return 'elevated'
        return 'normal'

    def _metric_trend(self, delta, scale):
        scale = max(float(scale), 1e-9)
        dz = float(delta) / scale
        if dz >= 2:
            return 'spike_up'
        if dz <= -2:
            return 'spike_down'
        if dz >= 0.5:
            return 'rising'
        if dz <= -0.5:
            return 'falling'
        return 'stable'

    def _count_bin(self, value):
        value = int(value)
        if value <= 0:
            return 'none'
        if value == 1:
            return 'one'
        if value <= 3:
            return 'few'
        if value <= 8:
            return 'some'
        if value <= 20:
            return 'many'
        return 'burst'

    def _window_count_bin(self, value):
        value = int(value)
        if value <= 0:
            return 'none'
        if value <= 2:
            return 'rare'
        if value <= 8:
            return 'short'
        if value <= 24:
            return 'medium'
        if value <= 64:
            return 'long'
        return 'sustained'

    def _severity_level(self, value):
        value = abs(float(value))
        if value >= 8:
            return 'extreme'
        if value >= 4:
            return 'high'
        if value >= self.abnormal_z:
            return 'elevated'
        return 'normal'

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
            if self.max_lines_per_file:
                df = df.head(int(self.max_lines_per_file))
            numeric = df.select_dtypes(include='number').copy()
            numeric = numeric.loc[:, ~numeric.columns.duplicated()]
            metric_cols = [c for c in numeric.columns if c != time_col and not str(c).startswith('time.')]
            time_values = pd.to_numeric(df[time_col], errors='coerce') if time_col in df.columns else pd.Series(range(len(df)), index=df.index)
            if inject_time is not None:
                baseline_mask = time_values < inject_time
            else:
                baseline_mask = pd.Series(False, index=df.index)
                baseline_mask.iloc[:max(1, int(len(df) * 0.3))] = True
            baseline = df.loc[baseline_mask, metric_cols].apply(pd.to_numeric, errors='coerce')
            if baseline.empty or baseline.dropna(how='all').empty:
                baseline = df[metric_cols].head(max(1, int(len(df) * 0.3))).apply(pd.to_numeric, errors='coerce')
            med = baseline.median(numeric_only=True)
            q75 = baseline.quantile(0.75, numeric_only=True)
            q25 = baseline.quantile(0.25, numeric_only=True)
            scale = (q75 - q25).replace(0, pd.NA).fillna(baseline.std(numeric_only=True)).replace(0, pd.NA).fillna(1.0)
            metric_values = df[metric_cols].apply(pd.to_numeric, errors='coerce')
            aligned_med_frame = med.reindex(metric_cols).fillna(metric_values.median(numeric_only=True)).astype(float)
            aligned_scale_frame = scale.reindex(metric_cols).fillna(1.0).astype(float).clip(lower=1e-9)
            z_frame = ((metric_values - aligned_med_frame) / aligned_scale_frame).replace([pd.NA, float('inf'), float('-inf')], 0).fillna(0)
            delta_frame = metric_values.diff().fillna(0)
            delta_z_frame = (delta_frame / aligned_scale_frame).replace([pd.NA, float('inf'), float('-inf')], 0).fillna(0)
            z_abs = z_frame.abs()
            delta_abs = delta_z_frame.abs()
            severity_frame = z_abs.where(z_abs >= delta_abs, delta_abs)
            service = fp.parent.parent.name
            run_id = fp.parent.name
            window_stats = {}
            for i, row in df.iterrows():
                ts = float(row[time_col]) if time_col in row else float(i)
                label = int(inject_time is not None and ts >= inject_time)
                window = int(i) // self.window_size
                session_id = f'{service}/{run_id}/window_{window:04d}'
                values = metric_values.loc[i].dropna()
                if values.empty:
                    messages = ['metrics heartbeat']
                else:
                    z = z_frame.loc[i].reindex(values.index).fillna(0)
                    delta = delta_frame.loc[i].reindex(values.index).fillna(0)
                    aligned_scale = aligned_scale_frame.reindex(values.index).fillna(1.0)
                    severity = severity_frame.loc[i].reindex(values.index).fillna(0)
                    sorted_severity = severity.sort_values(ascending=False)
                    abnormal = sorted_severity[sorted_severity >= self.abnormal_z]
                    if self.sparse_events:
                        top = (abnormal if not abnormal.empty else sorted_severity.head(1)).head(self.max_metric_tokens_per_row)
                    else:
                        top = sorted_severity.head(self.top_k)
                    max_metric = str(sorted_severity.index[0]) if len(sorted_severity) else 'metric'
                    max_value = float(sorted_severity.iloc[0]) if len(sorted_severity) else 0.0
                    dominant_type = self._metric_type(max_metric)
                    dominant_trend = self._metric_trend(delta[max_metric], aligned_scale[max_metric]) if max_metric in delta.index else 'stable'
                    messages = []
                    if self.row_features:
                        messages.extend([
                            f'row_severity_{self._severity_level(max_value)}',
                            f'row_abnormal_count_{self._count_bin(len(abnormal))}',
                            f'row_dominant_type_{self._safe_token(dominant_type, "type")}',
                            f'row_dominant_trend_{dominant_trend}',
                        ])
                    if self.sparse_events and abnormal.empty:
                        messages.append('metrics_all_normal')
                    for name in top.index:
                        if self.sparse_events and abnormal.empty:
                            continue
                        direction = 'above' if float(z[name]) > 0.5 else ('below' if float(z[name]) < -0.5 else 'near')
                        metric_type = self._metric_type(name)
                        messages.append(
                            f'{self._safe_token(name)} type_{self._safe_token(metric_type, "kind")} '
                            f'level_{self._severity_level(severity[name])} direction_{direction} '
                            f'trend_{self._metric_trend(delta[name], aligned_scale[name])}'
                        )
                    stats = window_stats.setdefault(session_id, {
                        'label': 0,
                        'service': service,
                        'source_file': str(fp),
                        'max_timestamp': 0,
                        'max_severity': 0.0,
                        'abnormal_rows': 0,
                        'type_counts': {},
                    })
                    stats['label'] = max(stats['label'], label)
                    stats['max_timestamp'] = max(stats['max_timestamp'], int(i) * 10 + len(messages))
                    stats['max_severity'] = max(stats['max_severity'], max_value)
                    if len(abnormal) > 0:
                        stats['abnormal_rows'] += 1
                        stats['type_counts'][dominant_type] = stats['type_counts'].get(dominant_type, 0) + 1
                for j, msg in enumerate(messages):
                    rows.append({
                        'session_id': session_id,
                        'timestamp': int(i) * 10 + j,
                        'raw_message': msg,
                        'label': label,
                        'service': service,
                        'source_file': str(fp),
                    })
            if self.window_features:
                for session_id, stats in window_stats.items():
                    type_counts = stats['type_counts']
                    dominant_type = max(type_counts.items(), key=lambda kv: kv[1])[0] if type_counts else 'none'
                    messages = [
                        f'window_max_severity_{self._severity_level(stats["max_severity"])}',
                        f'window_abnormal_rows_{self._window_count_bin(stats["abnormal_rows"])}',
                        f'window_dominant_type_{self._safe_token(dominant_type, "type")}',
                    ]
                    for metric_type, count in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
                        messages.append(f'window_type_{self._safe_token(metric_type, "kind")}_count_{self._count_bin(count)}')
                    for j, msg in enumerate(messages):
                        rows.append({
                            'session_id': session_id,
                            'timestamp': stats['max_timestamp'] + 1000 + j,
                            'raw_message': msg,
                            'label': int(stats['label']),
                            'service': stats['service'],
                            'source_file': stats['source_file'],
                        })
        return pd.DataFrame(rows)

def get_adapter(
    dataset_name,
    raw_dir,
    max_files=None,
    max_lines_per_file=None,
    max_runs_per_scenario=None,
    session_granularity=None,
    loghub_use_official_event_id=False,
    loghub_block_features=False,
    loghub_max_blocks=None,
    loghub_max_normal_blocks=None,
    loghub_max_anomaly_blocks=None,
    loghub_chunk_size=500000,
    loghub_rare_transition_max_count=2,
    loghub_transition_top_k=32,
    rcaeval_window_size=128,
    rcaeval_top_k=8,
    rcaeval_abnormal_z=2.0,
    rcaeval_sparse_events=False,
    rcaeval_row_features=True,
    rcaeval_window_features=False,
    rcaeval_max_metric_tokens_per_row=16,
):
    name = dataset_name.lower()
    if name in ['demo', 'generic']:
        return GenericDirectoryLogAdapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file)
    if name == 'lo2':
        return LO2Adapter(raw_dir, max_files=max_files, max_lines_per_file=max_lines_per_file, max_runs_per_scenario=max_runs_per_scenario, session_granularity=session_granularity or 'scenario')
    if name in ['loghub', 'loghub2', 'loghub-2.0']:
        return LoghubAdapter(
            raw_dir,
            max_files=max_files,
            max_lines_per_file=max_lines_per_file,
            use_official_event_id=loghub_use_official_event_id,
            block_features=loghub_block_features,
            max_blocks=loghub_max_blocks,
            max_normal_blocks=loghub_max_normal_blocks,
            max_anomaly_blocks=loghub_max_anomaly_blocks,
            chunk_size=loghub_chunk_size,
            rare_transition_max_count=loghub_rare_transition_max_count,
            transition_top_k=loghub_transition_top_k,
        )
    if name == 'rcaeval':
        return RCAEvalAdapter(
            raw_dir,
            max_files=max_files,
            max_lines_per_file=max_lines_per_file,
            window_size=rcaeval_window_size,
            top_k=rcaeval_top_k,
            abnormal_z=rcaeval_abnormal_z,
            sparse_events=rcaeval_sparse_events,
            row_features=rcaeval_row_features,
            window_features=rcaeval_window_features,
            max_metric_tokens_per_row=rcaeval_max_metric_tokens_per_row,
        )
    raise ValueError(f'Unknown dataset_name: {dataset_name}')
