import inspect
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import federated_main
import update
from federated_main import (
    _capture_loader_generator_states,
    _restore_loader_generator_states,
    _validate_ours_configuration,
)
from update import _preserve_feature_probe_state


def _assert_rng_state_equal(lhs, rhs):
    assert lhs[0] == rhs[0]
    assert np.array_equal(lhs[1][1], rhs[1][1])
    assert lhs[1][2:] == rhs[1][2:]
    assert torch.equal(lhs[2], rhs[2])


def test_feature_probe_is_observational():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.BatchNorm1d(4),
    )
    model.train()
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    dedicated_generator = torch.Generator().manual_seed(123)
    generator_before = dedicated_generator.get_state().clone()
    with _preserve_feature_probe_state(model):
        random.random()
        np.random.rand()
        torch.rand(3)
        torch.randperm(8, generator=dedicated_generator)
        model(torch.randn(8, 4))

    for name, value in model.named_buffers():
        assert torch.equal(value, buffers_before[name]), name
    rng_after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _assert_rng_state_equal(rng_before, rng_after)
    assert not torch.equal(dedicated_generator.get_state(), generator_before)


def test_feature_probe_switch_preserves_legacy_default():
    selector = getattr(update, "_feature_probe_state_context", None)
    assert selector is not None, "missing explicit feature-probe state switch"
    assert inspect.signature(update.LocalUpdate.update_weights_ours_debug).parameters[
        "preserve_feature_probe_state"
    ].default is False
    assert inspect.signature(federated_main.ours).parameters[
        "preserve_feature_probe_state"
    ].default is False

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.BatchNorm1d(4),
    )
    model.train()
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    with selector(model, preserve_feature_probe_state=False):
        random.random()
        np.random.rand()
        torch.rand(3)
        model(torch.randn(8, 4))

    assert any(
        not torch.equal(value, buffers_before[name])
        for name, value in model.named_buffers()
    ), "legacy mode must retain the historical BN-buffer side effect"
    rng_after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        _assert_rng_state_equal(rng_before, rng_after)
    except AssertionError:
        pass
    else:
        raise AssertionError("legacy mode must retain the historical RNG side effect")


def test_official_drivers_enable_probe_protection():
    driver_expectations = {
        "run_a_local_expand_office_dslr.py": "preserve_feature_probe_state=True",
        "run_b_server_expand_office_dslr.py": "preserve_feature_probe_state=True",
        "run_d_resampling_office_dslr.py": "preserve_feature_probe_state=True",
        "run_fragmentation_sweep_office_dslr.py": (
            'preserve_feature_probe_state=(_condition == "D")'
        ),
    }
    base = __import__("pathlib").Path(__file__).resolve().parent
    for filename, expected in driver_expectations.items():
        source = (base / filename).read_text(encoding="utf-8")
        assert expected in source, f"{filename} does not enable the agreed protection"


def test_feature_loader_routing_contract_is_preserved():
    update_source = inspect.getsource(update.LocalUpdate.update_weights_ours_debug)
    ours_source = inspect.getsource(federated_main.ours)
    assert "feature_loader if feature_loader is not None else train_loader" in update_source
    assert "feature_loader=client_feature_loader" in ours_source


def test_loader_generator_checkpoint_roundtrip():
    dataset = TensorDataset(torch.arange(12))
    generator = torch.Generator().manual_seed(321)
    loader = DataLoader(
        dataset,
        batch_size=3,
        sampler=RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
            generator=generator,
        ),
    )
    states = _capture_loader_generator_states([loader])
    list(loader)
    assert not torch.equal(generator.get_state(), states[0])
    _restore_loader_generator_states([loader], states, "test")
    assert torch.equal(generator.get_state(), states[0])


def test_configuration_validation():
    args = SimpleNamespace(num_users=2, domain_keyed_proto=False)
    try:
        _validate_ours_configuration(args, [object(), object()], True)
    except ValueError:
        pass
    else:
        raise AssertionError("mean+dispersion must require domain_keyed_proto")

    args.domain_keyed_proto = True
    try:
        _validate_ours_configuration(args, [object()], True)
    except ValueError:
        pass
    else:
        raise AssertionError("feature_loader_list length mismatch must fail")


if __name__ == "__main__":
    test_feature_probe_is_observational()
    test_feature_probe_switch_preserves_legacy_default()
    test_official_drivers_enable_probe_protection()
    test_feature_loader_routing_contract_is_preserved()
    test_loader_generator_checkpoint_roundtrip()
    test_configuration_validation()
    print("PASS: A/B experiment contracts")
