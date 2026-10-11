"""Choose the PyTorch training implementation from the loaded YAML."""

from openpi.training.config import TrainConfig


def train(config: TrainConfig) -> None:
    if config.mode == "rlt" and config.rlt.stage != "sft" and config.num_gpus != 1:
        raise ValueError("actor_critic and force use one process; set train.num_gpus: 1")
    if config.mode == "vla":
        from openpi.training import vla

        vla.main(config)
    elif config.rlt.stage == "sft":
        from openpi.training import rlt

        rlt.train(config)
    elif config.rlt.stage == "actor_critic":
        from openpi.rlt.train import train_from_config

        train_from_config(config)
    elif config.rlt.stage == "force":
        from openpi.rlt.force_rlt.train import train_from_config

        train_from_config(config)
    else:
        raise ValueError("rlt.stage must be sft, actor_critic or force")
