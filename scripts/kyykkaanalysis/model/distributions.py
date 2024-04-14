"""
Distribution functions.

This module provides implementations for functions that define distributions (e.g. log
densities and random number generators).
"""

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt
from pymc.math import exp, gt, log, lt, switch
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.math import gammainc, gammaincc, sum
from pytensor.tensor.variable import TensorVariable

DATA_INPUT_TYPE = (
    float | npt.NDArray[np.float64] | TensorVariable | Sequence[TensorVariable]
)


def podium_gamma_logp(
    value: DATA_INPUT_TYPE,
    k: float | TensorVariable,
    theta: float | TensorVariable,
) -> TensorVariable:
    """
    Log density of gamma distributed random variable with podium measurement error.

    Calculates the log probability density of a gamma distributed random variable that
    was measured with podium error. Podium error is an error distribution for two random
    variables measured with independent uniform error between [-1, 1] and rounded down.

    Parameters
    ----------
    value : DATA_INPUT_TYPE
        Value of random variables
    k : float or TensorVariable
        Shape parameter of gamma distribution
    theta : float or TensorVariable
        Mean of gamma distribution

    Returns
    -------
    TensorVariable
        Log density

    See Also
    --------
    The error model and the motivation for it is explained more thoroughly in the
    report stored in KyykkaAnalysis repo.
    """

    alpha = k
    beta = k / theta
    above_mean = gt(value, theta)

    value = value + expand_dims([-2, -1, 0, 1, 2, 3], -1)
    # Switch for numerical stability. If value is large, _gammainc is close to 1 and the
    # small differences vanish due to floating point accuracy
    densities = switch(
        above_mean,
        _gammaincc(alpha, beta * value)[::-1, :],
        _gammainc(alpha, beta * value),
    )

    # Differences of gamma distribution's CDF
    densities = densities[3:, :][::-1, :] - densities[:3, :]
    weights = expand_dims([5 / 9, 3 / 9, 1 / 9], -1)

    return log(sum(densities * weights, 0))


def _gammainc(k: float | TensorVariable, x: DATA_INPUT_TYPE) -> TensorVariable:
    return switch(lt(x, 0), 0, gammainc(k, x))


def _gammaincc(k: float | TensorVariable, x: DATA_INPUT_TYPE) -> TensorVariable:
    return switch(lt(x, 0), 0, gammaincc(k, x))


def podium_gamma_rng(
    k: float,
    theta: float,
    *,
    rng: np.random.RandomState | np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> npt.NDArray[np.int_]:
    """
    Random number generator for gamma random variables measured with podium error.

    Generates gamma distributed random variables that are measured with podium error.
    Podium error is an error distribution for two random variables measured with
    independent uniform error between [-1, 1] and rounded down.

    Parameters
    ----------
    k : float
        Shape parameter of gamma distribution
    theta : float
        Mean of gamma distribution
    rng : numpy.random.RandomState or numpy.random.Generator, optional
        Pseudo-random number generator. Defaults to numpy.random.default_rng()
    size : tuple of int, optional
        Output shape. Returns a single value by default

    Returns
    -------
    numpy.ndarray of int
        Random values

    See Also
    --------
    The error model and the motivation for it is explained more thoroughly in the
    report stored in KyykkaAnalysis repo.
    """

    if rng is None:
        rng = np.random.default_rng()

    draws = rng.gamma(k, theta / k, size=size)
    draws += (
        rng.multinomial(1, [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9], size=size).argmax(1) - 2
    )

    return np.floor(draws)


def podium_invgamma_logp(
    value: DATA_INPUT_TYPE,
    a: float | TensorVariable,
    theta: float | TensorVariable,
) -> TensorVariable:
    """
    Log density of inverse gamma distributed random variable with podium error.

    Calculates the log probability density of an inverse gamma distributed random
    variable that was measured with podium error. Podium error is an error distribution
    for two random variables measured with independent uniform error between [-1, 1] and
    rounded down.

    Parameters
    ----------
    value : DATA_INPUT_TYPE
        Value of random variables
    a : float or TensorVariable
        Transformed shape parameter of inverse gamma distribution. Shape parameter
        alpha = exp(-a) + 1
    theta : float or TensorVariable
        Mean of inverse gamma distribution

    Returns
    -------
    TensorVariable
        Log density

    See Also
    --------
    The error model and the motivation for it is explained more thoroughly in the
    report stored in KyykkaAnalysis repo.
    """

    alpha = exp(-a) + 1
    beta = theta * exp(-a)
    above_mean = gt(value, theta)

    value = value + expand_dims([-2, -1, 0, 1, 2, 3], -1)
    # Switch for numerical stability. If value is large, _gammaincc is close to 1 and
    # the small differences vanish due to floating point accuracy
    densities = switch(
        above_mean,
        _gammainc(alpha, beta / value)[::-1, :],
        _gammaincc(alpha, beta / value),
    )

    # Differences of inverse gamma distribution's CDF
    densities = densities[3:, :][::-1, :] - densities[:3, :]
    weights = expand_dims([5 / 9, 3 / 9, 1 / 9], -1)

    return log(sum(densities * weights, 0))


def podium_invgamma_rng(
    a: float,
    theta: float,
    *,
    rng: np.random.RandomState | np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> npt.NDArray[np.int_]:
    """
    RNG for inverse gamma random variables measured with podium error.

    Generates inverse gamma distributed random variables that are measured with podium
    error. Podium error is an error distribution for two random variables measured with
    independent uniform error between [-1, 1] and rounded down.

    Parameters
    ----------
    a : float
        Transformed shape parameter of inverse gamma distribution. Shape parameter
        alpha = exp(-a) + 1
    theta : float
        Mean of inverse gamma distribution
    rng : numpy.random.RandomState or numpy.random.Generator, optional
        Pseudo-random number generator. Defaults to numpy.random.default_rng()
    size : tuple of int, optional
        Output shape. Returns a single value by default

    Returns
    -------
    numpy.ndarray of int
        Random values

    See Also
    --------
    The error model and the motivation for it is explained more thoroughly in the
    report stored in KyykkaAnalysis repo.
    """

    if rng is None:
        rng = np.random.default_rng()

    # If x ~ gamma(alpha,beta) -> 1/x ~ inv-gamma(alpha,beta)
    draws = 1 / rng.gamma(np.exp(-a) + 1, np.exp(a) / theta, size=size)
    draws += (
        rng.multinomial(1, [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9], size=size).argmax(1) - 2
    )

    return np.floor(draws)
