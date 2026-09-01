import yaml
from pathlib import Path


def test_diffusion_config_loads():
    cfg = yaml.safe_load(Path('configs/diffusion.yaml').read_text())
    assert 'model' in cfg or 'backbone' in cfg or 'training' in cfg


def test_llm_config_loads():
    cfg = yaml.safe_load(Path('configs/llm.yaml').read_text())
    assert isinstance(cfg, dict)
