from pathlib import Path
BASE_DIR = Path('.')
config = {
    # 'data_dir': BASE_DIR / 'dataset/',
    'data_dir': BASE_DIR / 'dataset/word_level/',
    # 'data_dir': Path("/data1/nghind/workspaces/projects/albert/albert_server/dataset/word_level"),
    # 'data_dir': Path("/data1/nghind/workspaces/projects/albert_pytorch/dataset/corpus/train/shard"),
    # 'data_dir': Path("/data1/nghind/workspaces/projects/albert_pytorch/dataset/corpus/facebook/shard_128"),
    'log_dir': BASE_DIR / 'logs',
    'figure_dir': BASE_DIR / "figure",
    'checkpoint_dir': BASE_DIR / "checkpoints/word_level/albert_large_shareall",
    'result_dir': BASE_DIR / "result",
    'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_large_zh',
    'albert_config_path': BASE_DIR / 'configs/albert_config_large.json',
    # 'albert_vocab_path': BASE_DIR / 'dataset/vocab.txt',
    'albert_vocab_path': BASE_DIR / 'dataset/word_vocab.txt',
    # 'albert_vocab_path': BASE_DIR / 'dataset/facebook_32kVocab_bert.vocab',
    'pre_load_data': BASE_DIR / "preload_data",
    # 'pre_load_data': Path("/data1/nghind/workspaces/projects/albert_pytorch/preload_data/facebook")
    # 'pre_load_data': Path("/data1/nghind/workspaces/projects/albert_pytorch/preload_data/news")
}

