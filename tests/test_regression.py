"""Phase 1 - the reference values every regression milestone was checked against.

The five clean points are the anchor: any implementation of predict / mse /
gradients must reproduce 274.0 and (-108.0, -32.0) at m = c = 0. That check
caught a missing NumPy reduction at R6 and would have caught the R3 sign
inversion in one run.
"""

import numpy as np
import pytest

import regression_from_scratch as lists    # list-based, R1-R4
import training as np_impl                 # NumPy, R6 onward

CLEAN_X = [1.0, 2.0, 3.0, 4.0, 5.0]
CLEAN_Y = [10.0, 13.0, 16.0, 19.0, 22.0]


# --------------------------------------------------------------------------
# The list implementation (R1-R4)
# --------------------------------------------------------------------------

def test_list_predict_is_exact_on_clean_data():
    assert lists.predict(CLEAN_X, 3.0, 7.0) == CLEAN_Y


def test_list_mse_at_origin():
    preds = lists.predict(CLEAN_X, 0.0, 0.0)
    assert lists.MSE(CLEAN_Y, preds) == pytest.approx(274.0)


def test_list_mse_known_case():
    """Predictions [2,4] against targets [1,2]: residuals 1 and 2, squares 1 and
    4, sum 5, divided by 2."""
    assert lists.MSE([1.0, 2.0], [2.0, 4.0]) == pytest.approx(2.5)


def test_list_mse_is_symmetric_because_of_the_square():
    assert lists.MSE([1.0, 2.0], [2.0, 4.0]) == lists.MSE([2.0, 4.0], [1.0, 2.0])


def test_list_mse_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        lists.MSE([1.0, 2.0, 3.0], [1.0, 2.0])


def test_list_gradients_at_origin():
    preds = lists.predict(CLEAN_X, 0.0, 0.0)
    dm, dc = lists.gradients(CLEAN_Y, preds, CLEAN_X)
    assert dm == pytest.approx(-108.0)
    assert dc == pytest.approx(-32.0)


def test_list_predict_owns_its_output():
    """Regression test for the R2 bug: the function appended to a module-level
    list, so repeated calls accumulated instead of recomputing."""
    first = lists.predict(CLEAN_X, 3.0, 7.0)
    second = lists.predict(CLEAN_X, 3.0, 7.0)
    assert first == second == CLEAN_Y


# --------------------------------------------------------------------------
# The NumPy implementation (R6 onward) - must agree with the list version
# --------------------------------------------------------------------------

CX = np.array(CLEAN_X)
CY = np.array(CLEAN_Y)


def test_numpy_matches_list_on_the_five_point_reference():
    preds = np_impl.predict(CX, 0.0, 0.0)
    assert np_impl.mse(CY, preds) == pytest.approx(274.0)
    dm, dc = np_impl.gradients(CX, CY, preds)
    assert dm == pytest.approx(-108.0)
    assert dc == pytest.approx(-32.0)


def test_numpy_gradients_return_scalars_not_arrays():
    """Regression test for the R6 bug: the reduction was missing, so gradients
    came back as arrays of per-observation values instead of one number each."""
    preds = np_impl.predict(CX, 0.0, 0.0)
    dm, dc = np_impl.gradients(CX, CY, preds)
    assert np.isscalar(dm) or np.ndim(dm) == 0
    assert np.isscalar(dc) or np.ndim(dc) == 0


@pytest.mark.parametrize("m,c,expected_dm,expected_dc", [
    (3.0, 7.0, 0.0, 0.0),        # at the minimum on clean data, both are zero
    (0.0, 0.0, -108.0, -32.0),
    (2.0, 5.0, -34.0, -10.0),
    (5.0, 9.0, 56.0, 16.0),      # past the true values: both POSITIVE
])
def test_gradient_signs_and_magnitudes(m, c, expected_dm, expected_dc):
    dm, dc = np_impl.gradients(CX, CY, np_impl.predict(CX, m, c))
    assert dm == pytest.approx(expected_dm)
    assert dc == pytest.approx(expected_dc)


def test_analytical_gradients_match_finite_differences():
    """The check that would have caught the R3 sign inversion in one run."""
    for m, c in [(0.0, 0.0), (2.0, 5.0), (5.0, 9.0)]:
        gm, gc = np_impl.gradients(np_impl.x_train, np_impl.y_train,
                                   np_impl.predict(np_impl.x_train, m, c))
        nm, nc = np_impl.numerical_gradients(np_impl.x_train, np_impl.y_train, m, c)
        assert gm == pytest.approx(nm, abs=1e-4)
        assert gc == pytest.approx(nc, abs=1e-4)


# --------------------------------------------------------------------------
# The seeded dataset (R6)
# --------------------------------------------------------------------------

def test_dataset_shapes():
    assert np_impl.x.shape == (50,)
    assert np_impl.x_train.shape == (40,)
    assert np_impl.x_test.shape == (10,)


def test_train_and_test_partition_the_data():
    """Fancy indexing on a permutation: every point used once, none shared."""
    assert len(np_impl.x_train) + len(np_impl.x_test) == len(np_impl.x)
    combined = np.concatenate([np_impl.x_train, np_impl.x_test])
    assert np.allclose(np.sort(combined), np.sort(np_impl.x))


def test_noise_is_zero_mean_ish():
    """rng.normal is symmetric around zero, unlike the (i % 5) * 2 pattern that
    was tried first and had a mean of +4."""
    noise = np_impl.y - (np_impl.TRUE_M * np_impl.x + np_impl.TRUE_C)
    assert abs(noise.mean()) < 1.0


def test_training_recovers_the_known_parameters():
    """The R6 headline. Seeded, so this is exact, not approximate."""
    m, c = 0.0, 0.0
    for _ in range(20000):
        dm, dc = np_impl.gradients(np_impl.x_train, np_impl.y_train,
                                   np_impl.predict(np_impl.x_train, m, c))
        m -= 0.01 * dm
        c -= 0.01 * dc
    assert m == pytest.approx(3.0707, abs=1e-4)
    assert c == pytest.approx(6.2686, abs=1e-4)
