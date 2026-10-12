import math

import numpy as np
import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from saev.utils.statistics import calc_batch_entropy, cosine_sums


def _tensor(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.int64)


def test_calc_batch_entropy_uniform_distribution() -> None:
    example_idx = _tensor([0, 0, 1, 1])
    token_idx = _tensor([0, 1, 0, 1])

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=2,
        content_tokens_per_example=2,
    )

    assert math.isclose(results["loader/example_entropy"], math.log(2.0), rel_tol=1e-8)
    assert math.isclose(results["loader/token_entropy"], math.log(2.0), rel_tol=1e-8)
    assert math.isclose(results["loader/example_entropy_normalized"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/token_entropy_normalized"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/example_coverage"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/token_coverage"], 1.0, rel_tol=1e-8)


def test_calc_batch_entropy_handles_skew_and_support() -> None:
    example_idx = _tensor([0, 0, 1, 1, 1, 1])
    token_idx = _tensor([0, 0, 0, 0, 0, 0])

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=3,
        content_tokens_per_example=5,
    )

    expected_example_entropy = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    assert math.isclose(
        results["loader/example_entropy"], expected_example_entropy, rel_tol=1e-8
    )
    assert results["loader/token_entropy"] == pytest.approx(0.0, rel=1e-8, abs=1e-12)
    assert math.isclose(
        results["loader/example_entropy_normalized"],
        expected_example_entropy / math.log(3.0),
        rel_tol=1e-8,
    )
    assert results["loader/token_entropy_normalized"] == pytest.approx(
        0.0, rel=1e-8, abs=1e-12
    )
    assert results["loader/example_coverage"] == pytest.approx(
        2 / 3, rel=1e-8, abs=1e-12
    )
    assert results["loader/token_coverage"] == pytest.approx(1 / 5, rel=1e-8, abs=1e-12)


def test_calc_batch_entropy_accepts_numpy_inputs() -> None:
    example_idx = np.array([4, 5, 4, 5, 6], dtype=np.int32)
    token_idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=5,
        content_tokens_per_example=8,
    )

    assert results["loader/example_coverage"] == pytest.approx(
        3 / 5, rel=1e-8, abs=1e-12
    )
    assert results["loader/token_coverage"] == pytest.approx(3 / 8, rel=1e-8, abs=1e-12)


def test_calc_batch_entropy_validates_inputs() -> None:
    with pytest.raises(ValueError):
        calc_batch_entropy(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            n_examples=1,
            content_tokens_per_example=1,
        )

    with pytest.raises(TypeCheckError):
        calc_batch_entropy(
            _tensor([0]),
            torch.empty(0, dtype=torch.int64),
            n_examples=1,
            content_tokens_per_example=1,
        )

    with pytest.raises(ValueError):
        calc_batch_entropy(
            _tensor([0]),
            _tensor([0]),
            n_examples=0,
            content_tokens_per_example=1,
        )

    with pytest.raises(ValueError):
        calc_batch_entropy(
            _tensor([0]),
            _tensor([0]),
            n_examples=1,
            content_tokens_per_example=0,
        )


def test_cosine_sums_perfect_reconstruction() -> None:
    x = torch.randn(5, 4)
    unc, cen = cosine_sums(x, x.clone(), x.mean(0))
    assert unc.item() == pytest.approx(5.0)
    assert cen.item() == pytest.approx(5.0)


def test_cosine_sums_ignores_scale_when_mean_is_zero() -> None:
    x = torch.randn(6, 4)
    x = x - x.mean(0)
    unc, cen = cosine_sums(x, 0.5 * x, torch.zeros(4))
    assert unc.item() == pytest.approx(6.0)
    assert cen.item() == pytest.approx(6.0)


def test_cosine_sums_centering_changes_the_answer_when_mean_is_nonzero() -> None:
    mean = torch.tensor([10.0, 0.0])
    x = torch.tensor([[11.0, 1.0], [9.0, -1.0]])
    x_hat = torch.tensor([[11.0, -1.0], [9.0, 1.0]])
    unc, cen = cosine_sums(x, x_hat, mean)
    # Uncentered vectors are nearly parallel (both close to the mean direction); centered deviations are orthogonal-ish.
    assert unc.item() / 2 > 0.97
    assert cen.item() / 2 == pytest.approx(0.0, abs=1e-9)


def test_cosine_sums_orthogonal() -> None:
    x = torch.tensor([[1.0, 0.0]])
    x_hat = torch.tensor([[0.0, 1.0]])
    unc, cen = cosine_sums(x, x_hat, torch.zeros(2))
    assert unc.item() == pytest.approx(0.0, abs=1e-12)
    assert cen.item() == pytest.approx(0.0, abs=1e-12)


def test_cosine_sums_returns_float64_scalars() -> None:
    unc, cen = cosine_sums(torch.randn(3, 4), torch.randn(3, 4), torch.zeros(4))
    assert unc.dtype == torch.float64 and unc.shape == ()
    assert cen.dtype == torch.float64 and cen.shape == ()


def test_cosine_sums_rejects_shape_mismatch() -> None:
    with pytest.raises(TypeCheckError):
        cosine_sums(torch.randn(3, 4), torch.randn(3, 5), torch.zeros(4))
