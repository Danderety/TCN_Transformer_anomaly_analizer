from collections import Counter
from pathlib import Path
import json

class EventVocab:
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>'):
        self.pad_token = pad_token; self.unk_token = unk_token
        self.event_to_id = {pad_token: 0, unk_token: 1}
        self.id_to_event = {0: pad_token, 1: unk_token}

    def build(self, templates, min_freq=1):
        for template, freq in Counter(templates).items():
            if freq >= min_freq and template not in self.event_to_id:
                idx = len(self.event_to_id)
                self.event_to_id[template] = idx
                self.id_to_event[idx] = template

    def encode(self, template):
        return self.event_to_id.get(template, self.event_to_id[self.unk_token])

    def decode(self, event_id):
        return self.id_to_event.get(int(event_id), self.unk_token)

    def __len__(self):
        return len(self.event_to_id)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'event_to_id': self.event_to_id, 'id_to_event': self.id_to_event}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        v.event_to_id = {str(k): int(val) for k, val in data['event_to_id'].items()}
        v.id_to_event = {int(k): val for k, val in data['id_to_event'].items()}
        return v
