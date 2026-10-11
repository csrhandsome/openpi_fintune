"""CPU/Gloo regressions for the actual tiny Pi0 and readout training loops."""

import dataclasses
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.rlinf.model import IMAGE_KEYS
from openpi.models_pytorch.rlinf.model import Observation
from openpi.tests.models_pytorch.checkpoint_alignment_test import tiny_backbone
from openpi.training import config as train_config
from openpi.training import rlt
from openpi.training import vla


def _config(directory, mode, alpha):
    config = train_config.get_config("openpi_rlt" if mode == "rlt" else "openpi")
    return dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            is_lora=True,
            lora_rank=2,
            dtype="float32",
            action_horizon=2,
        ),
        batch_size=4,
        micro_batch_size=1,
        num_gpus=2,
        num_workers=0,
        num_train_steps=2,
        save_interval=10,
        wandb_enabled=False,
        pytorch_training_precision="float32",
        requires_pytorch_weights=False,
        pytorch_weight_path=None,
        checkpoint_base_dir=str(directory / "vla-output"),
        lr_schedule=dataclasses.replace(config.lr_schedule, warmup_steps=0),
        rlt=dataclasses.replace(
            config.rlt,
            vla_checkpoint=str(directory / "base"),
            output_dir=str(directory / "full"),
            alpha=alpha,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            max_prefix_len=16,
            device="auto",
        ),
    )


def _count_reduce(state, bucket):
    state.append(1)
    buffer = bucket.buffer()
    dist.all_reduce(buffer)
    buffer.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


def _worker(rank, directory, mode, alpha):
    directory = Path(directory)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        tiny_backbone.__wrapped__(patch)
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(rlt, "validate_dataset", lambda _: None)
        patch.setattr(vla, "validate_dataset", lambda _: None)
        patch.setattr(
            train_config.LeRobotFrankaDataConfig,
            "create",
            lambda *args: SimpleNamespace(norm_stats={"state": object()}),
        )
        torch.manual_seed(12 + rank)
        observation = Observation(
            images={key: torch.randn(1, 224, 224, 3) for key in IMAGE_KEYS},
            image_masks={key: torch.ones(1, dtype=torch.bool) for key in IMAGE_KEYS},
            state=torch.randn(1, 32),
            tokenized_prompt=torch.ones(1, 2, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        actions = torch.randn(1, 2, 32)
        batches = [(observation, actions)]
        patch.setattr(rlt.data_loader, "create_data_loader", lambda *args, **kwargs: batches)
        patch.setattr(vla, "build_datasets", lambda *args, **kwargs: (batches, train_config.DataConfig()))
        wrappers = []
        ddp_class = torch.nn.parallel.DistributedDataParallel

        class CountingDDP(ddp_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reductions = []
                self.register_comm_hook(self.reductions, _count_reduce)
                wrappers.append(self)

        patch.setattr(torch.nn.parallel, "DistributedDataParallel", CountingDDP)
        config = _config(directory, mode, alpha)
        runs = [("full", config)]
        if mode == "rlt":
            first = dataclasses.replace(
                config,
                num_train_steps=1,
                rlt=dataclasses.replace(config.rlt, output_dir=str(directory / "resumed")),
            )
            runs.extend([("first", first), ("resumed", dataclasses.replace(first, num_train_steps=2, resume=True))])
        for label, settings in runs:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{directory}/rendezvous-{label}",
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=90),
            )
            try:
                if mode == "rlt":
                    rlt.train(settings)
                else:
                    vla.train_loop(settings)
                wrapper = wrappers[-1]
                expected_steps = 2 if label == "full" else 1
                assert len(wrapper.reductions) == expected_steps
                torch.save(wrapper.module.state_dict(), directory / f"{label}-rank-{rank}.pt")
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()


@pytest.mark.parametrize(("mode", "alpha"), [("rlt", 0.0), ("rlt", 1.0), ("vla", 1.0)])
def test_two_rank_training_and_resume(tmp_path, tiny_backbone, mode, alpha):
    del tiny_backbone
    config = _config(tmp_path, mode, alpha)
    (tmp_path / "base/assets").mkdir(parents=True)
    torch.manual_seed(12)
    base = PI0Pytorch(dataclasses.replace(config.model, is_lora=False))
    save_file(base.state_dict(), tmp_path / "base/model.safetensors")
    mp.spawn(_worker, args=(str(tmp_path), mode, alpha), nprocs=2, join=True)
    rank_states = [torch.load(tmp_path / f"full-rank-{rank}.pt", weights_only=True) for rank in range(2)]
    for name, value in rank_states[0].items():
        torch.testing.assert_close(rank_states[1][name], value, rtol=0, atol=0)
    if mode == "vla":
        assert (config.checkpoint_dir / "2/model.safetensors").is_file()
        assert not (config.checkpoint_dir / "1").exists()
        return
    full = torch.load(tmp_path / "full/2/training_state.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/2/training_state.pt", weights_only=True)
    for rank in range(2):
        state = torch.load(tmp_path / f"resumed-rank-{rank}.pt", weights_only=True)
        for name, value in rank_states[rank].items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        expected_rng, actual_rng = full["rng_by_rank"][rank], resumed["rng_by_rank"][rank]
        torch.testing.assert_close(expected_rng["rng"], actual_rng["rng"], rtol=0, atol=0)
        torch.testing.assert_close(expected_rng["numpy_rng"][1], actual_rng["numpy_rng"][1], rtol=0, atol=0)
    assert not torch.equal(full["rng_by_rank"][0]["rng"], full["rng_by_rank"][1]["rng"])
    for module in ("vla", "readout"):
        for name, expected in full["ema"][module].items():
            torch.testing.assert_close(resumed["ema"][module][name], expected, rtol=0, atol=0)
    if alpha == 0:
        assert all(
            torch.equal(base.state_dict()[name], value)
            for name, value in full["vla"].items()
            if name in base.state_dict()
        )
    else:
        assert any(
            name.endswith("lora_B.default.weight") and value.count_nonzero() for name, value in full["vla"].items()
        )
