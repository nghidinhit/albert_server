from pathlib import Path
BASE_DIR = Path('.')
config = {
    'data_dir': BASE_DIR / 'dataset/',
    'log_dir': BASE_DIR / 'logs',
    'figure_dir': BASE_DIR / "figure",
    'checkpoint_dir': BASE_DIR / "checkpoints",
    'result_dir': BASE_DIR / "result",
    'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_xlarge_zh',
    'albert_config_path': BASE_DIR / 'configs/albert_config_xlarge.json',
    'albert_vocab_path': BASE_DIR / 'dataset/vocab.txt',
    'pre_load_data': BASE_DIR / "preload_data"
}

