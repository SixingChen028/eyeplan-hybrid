import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from modules.a2c import load_jax_params, save_jax_params
from modules.environment_compat import ENVIRONMENT_COMPAT_VERSION, PARAMS_FORMAT_VERSION


def test_save_load_jax_params_round_trips_versioned_payload(tmp_path: Path):
    params = {"w": jnp.asarray([1.0, 2.0]), "b": jnp.asarray(0.5)}
    path = tmp_path / "net_jax.p"

    save_jax_params(params, str(path))
    loaded = load_jax_params(str(path))

    np.testing.assert_allclose(np.asarray(loaded["w"]), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(loaded["b"]), np.asarray(0.5))

    with path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["params_format_version"] == PARAMS_FORMAT_VERSION
    assert payload["environment_compat_version"] == ENVIRONMENT_COMPAT_VERSION
    assert "params" in payload


def test_load_jax_params_rejects_unversioned_payload_by_default(tmp_path: Path):
    path = tmp_path / "legacy_net_jax.p"
    with path.open("wb") as file:
        pickle.dump({"w": np.asarray([1.0])}, file)

    with pytest.raises(ValueError, match="missing environment compatibility metadata"):
        load_jax_params(str(path))


def test_load_jax_params_allows_unversioned_payload_when_requested(tmp_path: Path):
    path = tmp_path / "legacy_net_jax.p"
    with path.open("wb") as file:
        pickle.dump({"w": np.asarray([1.0])}, file)

    loaded = load_jax_params(str(path), allow_unversioned=True)

    np.testing.assert_allclose(np.asarray(loaded["w"]), np.asarray([1.0]))


def test_load_jax_params_rejects_environment_compat_mismatch(tmp_path: Path):
    path = tmp_path / "net_jax.p"
    payload = {
        "params_format_version": PARAMS_FORMAT_VERSION,
        "environment_compat_version": ENVIRONMENT_COMPAT_VERSION + 1,
        "params": {"w": np.asarray([1.0])},
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)

    with pytest.raises(ValueError, match="Environment compatibility version mismatch"):
        load_jax_params(str(path))


def test_load_jax_params_rejects_metadata_compat_mismatch(tmp_path: Path):
    path = tmp_path / "net_jax.p"
    save_jax_params({"w": jnp.asarray([1.0])}, str(path))

    with pytest.raises(ValueError, match="run metadata"):
        load_jax_params(str(path), expected_environment_compat_version=ENVIRONMENT_COMPAT_VERSION + 1)
