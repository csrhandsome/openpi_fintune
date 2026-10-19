import pytest
import yaml

from openpi.training import config


@pytest.fixture
def train_yaml():
    """Read expected configuration values independently of the config parser."""

    def read(name="openpi"):
        return yaml.safe_load((config.REPO_ROOT / "config/train" / f"{name}.yaml").read_text())

    return read
