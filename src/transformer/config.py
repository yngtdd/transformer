def load_config():
    """Simple store for hyperparameters"""
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_target": "it",
        "model_folder": "weights",
        "model_filename": "transformer_model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transfomer_model",
    }
