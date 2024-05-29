from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "validation_batch_size": 1,
        "num_epochs": 13,
        "lr" : 10e-4,
        "seq_len" : 350,
        "d_model" : 512,
        "lang_src" : "en",
        "lang_tgt" : "it",
        "model_folder": 'weight',
        "model_basename": 'tmodel2_',
        "preload": "02",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs3/tmodel"
    }

def get_weight_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder/ model_filename)