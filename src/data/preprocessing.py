import re

def clean_log_message(message, lowercase=False, remove_timestamps=True, remove_numbers=False):
    text = str(message).strip()
    if lowercase:
        text = text.lower()
    if remove_timestamps:
        text = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[\.,]?\d*', '<TIME>', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}[\.,]?\d*', '<TIME>', text)
    if remove_numbers:
        text = re.sub(r'\b\d+\b', '<NUM>', text)
    return re.sub(r'\s+', ' ', text)

def simple_template_parser(message):
    text = str(message)
    text = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<UUID>', text)
    text = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>', text)
    text = re.sub(r'0x[a-fA-F0-9]+', '<HEX>', text)
    text = re.sub(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+', '<EMAIL>', text)
    text = re.sub(r'(/[A-Za-z0-9_\-\.]+)+', '<PATH>', text)
    text = re.sub(r'@[0-9a-fA-F]{6,}\b', '@<ID>', text)
    text = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<ID>', text)
    text = re.sub(r'\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{16,}\b', '<ID>', text)
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    return re.sub(r'\s+', ' ', text).strip()
