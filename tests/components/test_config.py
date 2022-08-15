from cool_project.config import ModelConfig


def test_example():
    model_config = ModelConfig()
    assert model_config.in_channels == 1 * 28 * 28
    assert model_config.hidden_size == 64
    assert model_config.num_classes == 10
