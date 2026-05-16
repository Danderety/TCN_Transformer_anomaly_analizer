from src.models.tcn_ae import TCNAutoencoder
from src.models.tcn_transformer_ae import TCNTransformerAutoencoder

def build_model(model_name, config):
    m = config['model']; d = config['data']; vocab_size = m['vocab_size']
    if vocab_size is None: raise ValueError('model.vocab_size is None. Run 01_prepare_data.py and then use processed used_config.yaml')
    if model_name == 'tcn_ae':
        return TCNAutoencoder(vocab_size, m['d_model'], m['tcn_layers'], m['dropout'], d['pad_id'])
    if model_name == 'tcn_transformer_ae':
        return TCNTransformerAutoencoder(vocab_size, m['d_model'], m['tcn_layers'], m['transformer_layers'], m['nhead'], m['dim_feedforward'], m['dropout'], d['pad_id'])
    raise ValueError(model_name)
