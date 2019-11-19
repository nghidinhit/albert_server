from pathlib import Path
BASE_DIR = Path('.')
config = {
    'corpus_path': '/media/nghind/DATA1/data/text_corpus/News_corpus_VuQuocBinh/corpus-cate-csv-2019-24-03/news.txt', 
    'data_dir': BASE_DIR / 'dataset/',
    'log_dir': BASE_DIR / 'outputs/logs',
    'figure_dir': BASE_DIR / "outputs/figure",
    'outputs': BASE_DIR / 'outputs',
    'checkpoint_dir': BASE_DIR / "outputs/checkpoints",
    'result_dir': BASE_DIR / "outputs/result",
    'bert_dir':BASE_DIR / 'pretrain/pytorch/albert_xlarge_zh',
    'albert_config_path': BASE_DIR / 'configs/albert_config_xlarge.json',
    'albert_vocab_path': BASE_DIR / 'dataset/vocab.txt',
    'pre_load_data': BASE_DIR / "outputs/preload_data"
}

