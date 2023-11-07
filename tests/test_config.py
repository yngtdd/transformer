from transformer.config import load_config


def test_default_batch_size():
    """Test that the default batch size is smol"""
    config = load_config()
    assert config["batch_size"] == 8
