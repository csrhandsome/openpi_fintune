import pytest
import torch

from force_encoder.model import ImageJEPA
from force_encoder.model import load_encoder
from force_encoder.model import sample_masks


@pytest.fixture
def model():
    return ImageJEPA(
        image_size=32, patch_size=8, embed_dim=24, depth=1, num_heads=3, predictor_depth=1, predictor_dim=24
    )


def test_masks_exclude_all_targets_and_bound_overlap():
    for seed in range(10):
        masks = sample_masks(14, generator=torch.Generator().manual_seed(seed))
        context = set(masks.context.tolist())
        assert context
        assert len(masks.targets) == 4
        for index, target in enumerate(masks.targets):
            target_set = set(target.tolist())
            assert not context & target_set
            # Every target fills its rectangular bounding box.
            rows, columns = target // 14, target % 14
            assert target.numel() == (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
            for previous in masks.targets[:index]:
                assert len(target_set & set(previous.tolist())) / min(target.numel(), previous.numel()) <= 0.25


def test_mask_reproducibility_and_impossible_constraints():
    first = sample_masks(4, generator=torch.Generator().manual_seed(1))
    second = sample_masks(4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(first.context, second.context)
    assert all(torch.equal(left, right) for left, right in zip(first.targets, second.targets, strict=True))
    with pytest.raises(ValueError, match="Could not place"):
        sample_masks(2, local_scale=(1, 1), num_local_targets=1, num_global_targets=0)


def test_loss_gradients_and_frozen_deterministic_teacher(model):
    model.train()
    assert not model.target_encoder.training
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(images, generator=torch.Generator().manual_seed(8))
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.predictor.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.context_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    # Patch projection has no overlap: excluded target pixels cannot influence context.
    masks = model.sample_masks(torch.Generator().manual_seed(8))
    pixel_grad = images.grad.reshape(2, 3, 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).reshape(2, 16, -1)
    all_targets = torch.unique(torch.cat(masks.targets))
    assert torch.count_nonzero(pixel_grad[:, all_targets]) == 0
    assert torch.equal(model.target_encoder(images.detach()), model.target_encoder(images.detach()))


def test_teacher_ema(model):
    teacher_before = next(model.target_encoder.parameters()).clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1)
    model.update_teacher(0.75)
    assert torch.allclose(next(model.target_encoder.parameters()), teacher_before + 0.25)
    with pytest.raises(ValueError, match="momentum"):
        model.update_teacher(1.5)


def test_inference_checkpoint_roundtrip(model, tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, checkpoint)
    encoder = load_encoder(checkpoint)
    images = torch.randn(2, 3, 32, 32)
    assert encoder(images).shape == (2, 24)
    assert encoder.forward_features(images).shape == (2, 16, 24)
    assert torch.equal(encoder(images), model.encode(images))
    assert torch.equal(encoder.forward_features(images), model.encode(images, pool=False))
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
