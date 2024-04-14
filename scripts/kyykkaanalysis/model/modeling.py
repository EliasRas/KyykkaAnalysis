"""
Constructing play time models.

This module constructs kyykkä play time models and provides containers for them with
functionalities for interacting with the models.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Self

import numpy as np
import pymc as pm
from arviz import ELPDData, InferenceData, ess, loo, loo_pit, psislw, summary
from numpy import typing as npt
from pymc.distributions.shape_utils import Shape
from pymc.math import exp, floor, gt, log, lt, switch
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.math import gammainc, gammaincc
from pytensor.tensor.variable import TensorVariable
from xarray import DataArray, Dataset, merge

from ..data.data_classes import ModelData

DATA_INPUT_TYPE = (
    float | npt.NDArray[np.float64] | TensorVariable | Sequence[TensorVariable]
)


class ModelType(Enum):
    """Type of throw time model."""

    GAMMA = 1
    """
    Uses a gamma distribution for raw data and podium error model
    """
    NAIVE = 2
    """
    Uses a gamma distribution for raw data and floor rounding error model
    """
    INVGAMMA = 3
    """
    Uses an inverse gamma distribution for raw data and podium error model
    """
    NAIVEINVGAMMA = 4
    """
    Uses an inverse gamma distribution for raw data and floor rounding error model
    """


class ThrowTimeModel:
    """
    Container for throw time model.

    Stores kyykkä throw time model. Provides methods for sampling from the
    distributions defined by the model, calculating cross-validation results,
    and processing samples.

    Attributes
    ----------
    data : ModelData
        Data for the model
    model_type : ModelType
        The type of model used
    model : pymc.Model
        Throw time model
    """

    def __init__(
        self, data: ModelData, *, model_type: ModelType = ModelType.GAMMA
    ) -> None:
        """
        Container for throw time model.

        Parameters
        ----------
        data : ModelData
            Data for the model
        model_type : ModelType, default GAMMA
            Type of model to construct
        """

        self.data = data
        self.model_type = model_type
        match model_type:
            case ModelType.GAMMA:
                self.model = gamma_throw_model(data)
            case ModelType.NAIVE:
                self.model = gamma_throw_model(data, naive=True)
            case ModelType.INVGAMMA:
                self.model = invgamma_throw_model(data)
            case ModelType.NAIVEINVGAMMA:
                self.model = invgamma_throw_model(data, naive=True)

    def sample_prior(self, *, sample_count: int = 500) -> Dataset:
        """
        Sample from the prior predictive distribution.

        Parameters
        ----------
        sample_count : int, default 500
            Number of samples to draw

        Returns
        -------
        xarray.Dataset
            Prior samples
        """

        with self.model:
            samples = pm.sample_prior_predictive(samples=sample_count)

        samples = merge([samples.prior, samples.prior_predictive])
        samples["y"] = samples["y"].astype(int)

        return samples

    def sample(  # noqa: PLR0913
        self,
        *,
        sample_count: int = 1000,
        tune_count: int = 1000,
        chain_count: int = 4,
        parallel_count: int = 1,
        thin: bool = True,
    ) -> InferenceData:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        sample_count : int, default 1000
            Number of samples to draw in each chain
        tune_count : int, default 1000
            Number of iterations run while tuning the sampler
        chain_count : int, default 4
            Number of chains to run
        parallel_count : int, default 1
            Number of parallel chains to run
        thin : bool, default True
            Whether to thin the chains to reduce autocorrelation

        Returns
        -------
        arviz.InferenceData
            Posterior samples
        """

        with self.model:
            starting_point = self._starting_point()
            samples = pm.sample(
                nuts_sampler="blackjax",
                draws=sample_count,
                tune=tune_count,
                chains=chain_count,
                cores=parallel_count,
                initvals=starting_point,
                init="adapt_diag",
            )

        if thin:
            return self.thin(samples)

        return samples

    def _starting_point(self) -> dict[str, npt.NDArray[np.float64]]:
        starting_point = {
            "mu_interval__": np.array(np.log(28)),
            "sigma_log__": np.array(np.log(11)),
            "o_log__": np.array(np.log(1)),
            "theta_interval__": np.ones(len(self.model.coords["players"])) * np.log(28),
        }

        match self.model_type:
            case ModelType.GAMMA | ModelType.NAIVE:
                starting_point["k_interval__"] = np.array(np.log(2))
            case ModelType.INVGAMMA | ModelType.NAIVEINVGAMMA:
                starting_point["a"] = np.array(-2)

        return starting_point

    def sample_posterior_predictive(self, posterior_sample: Dataset) -> Dataset:
        """
        Sample from the posterior predictive distribution.

        Parameters
        ----------
        posterior_sample : xarray.Dataset
            Posterior samples

        Returns
        -------
        xarray.Dataset
            Posterior predictive samples
        """

        with self.model:
            samples = pm.sample_posterior_predictive(posterior_sample)

        samples = samples.posterior_predictive
        samples["y"] = samples["y"].astype(int)

        return samples

    def psis_loo(
        self, posterior_samples: Dataset, posterior_predictive: Dataset
    ) -> ELPDData:
        """
        Carry out Pareto smoothed importance sampling leave-one-out cross-validation.

        Parameters
        ----------
        posterior_samples : xarray.Dataset
            Posterior samples
        posterior_predictive : xarray.Dataset
            Posterior predictive samples

        Returns
        -------
        arviz.ELPDData
            Cross-validation results
        """

        posterior = InferenceData(posterior=posterior_samples)
        with self.model:
            pm.compute_log_likelihood(posterior)

        loo_result = loo(posterior, pointwise=True)

        y_hat = posterior_predictive["y"].stack(__sample__=("chain", "draw"))
        pit = loo_pit(
            y=self.model.throw_times.container.data,
            y_hat=y_hat,
            log_weights=self._psis_weights(posterior_samples, posterior.log_likelihood),
        )
        loo_result.pit = DataArray(
            data=pit, dims=["throws"], coords={"throws": loo_result.loo_i.throws}
        )

        return loo_result

    def _psis_weights(
        self, posterior_samples: Dataset, log_likelihood: Dataset
    ) -> None:
        log_likelihood = log_likelihood["y"].stack(__sample__=("chain", "draw"))

        if posterior_samples["chain"].size > 1:
            sample_count = log_likelihood.__sample__.size
            ess_p = ess(posterior_samples, method="mean")
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
                / sample_count
            )
        else:
            reff = 1

        weights = psislw(-log_likelihood, reff=reff)[0].values

        return weights

    def change_observations(self, *, y: npt.NDArray[np.int_] | None = None) -> Self:
        """
        Change the observed data in the model.

        Parameters
        ----------
        y : numpy.ndarray of int or None
            Throw times

        Returns
        -------
        ThrowTimeModel
            Model with updated observations
        """

        if y is not None:
            with self.model:
                y = y.flatten()

                # Ensure that measurement error bounds are within relevant distribution
                # domain. Otherwise gradient calculation fails
                match self.model_type:
                    case ModelType.GAMMA:
                        y[y < 3] = 3  # noqa: PLR2004
                    case ModelType.NAIVE:
                        y[y < 1] = 1
                    case ModelType.INVGAMMA:
                        y[y < 3] = 3  # noqa: PLR2004
                    case ModelType.NAIVEINVGAMMA:
                        y[y < 1] = 1

                pm.set_data({"throw_times": y})

        return self

    @staticmethod
    def thin(samples: InferenceData) -> InferenceData:
        """
        Thin the chains to reduce autocorrelations.

        Parameters
        ----------
        samples : arviz.InferenceData
            Samples from the chains

        Returns
        -------
        arviz.InferenceData
            Thinned chains
        """

        sample_count = (
            samples.posterior.sizes["chain"] * samples.posterior.sizes["draw"]
        )
        max_ess = summary(samples)["ess_bulk"].max()
        if max_ess > sample_count:
            # Get rid of negative autocorrelations à la NUTS
            samples.posterior = samples.posterior.thin({"draw": 2})
            samples.sample_stats = samples.sample_stats.thin({"draw": 2})

        sample_count = (
            samples.posterior.sizes["chain"] * samples.posterior.sizes["draw"]
        )
        min_ess = summary(samples)["ess_bulk"].min()
        subsample_step = int(sample_count // min_ess)
        if subsample_step > 1:
            # Get rid of autocorrelation
            samples.posterior = samples.posterior.thin({"draw": subsample_step})
            samples.sample_stats = samples.sample_stats.thin({"draw": subsample_step})

        return samples

    @staticmethod
    def thin_posterior(samples: Dataset) -> Dataset:
        """
        Thin the chains to reduce autocorrelations.

        Parameters
        ----------
        samples : xarray.Dataset
            Samples from the chains

        Returns
        -------
        xarray.Dataset
            Thinned chains
        """

        sample_count = samples.sizes["chain"] * samples.sizes["draw"]
        max_ess = summary(samples)["ess_bulk"].max()
        if max_ess > sample_count:
            # Get rid of negative autocorrelations à la NUTS
            samples = samples.thin({"draw": 2})

        sample_count = samples.sizes["chain"] * samples.sizes["draw"]
        min_ess = summary(samples)["ess_bulk"].min()
        subsample_step = int(sample_count // min_ess)
        if subsample_step > 1:
            # Get rid of autocorrelation
            samples = samples.thin({"draw": subsample_step})

        return samples

    @property
    def observed_variables(self) -> set[str]:
        """
        Names of the observed variables.

        Returns
        -------
        set of str
            Observed variables
        """

        return {"y"}

    @property
    def dataset(self) -> Dataset:
        """
        Data for the model as xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            Data for the model
        """

        data = Dataset(
            {
                "player": (["throws"], self.model.player.container.data),
                "is_first": (
                    ["throws"],
                    self.model.is_first.container.data.astype(bool),
                ),
                "y": (["throws"], self.model.throw_times.container.data.astype(int)),
            },
            coords={"throws": np.array(self.model.coords["throws"])},
        )

        return data


def gamma_throw_model(data: ModelData, *, naive: bool = False) -> pm.Model:
    """
    Construct a model for throw times.

    Constructs a model for throw times which uses gamma distribution for the underlying
    distribution of noise free throw times. The error model is either floor rounding
    or a more complex distribution based on error distribution of timestamps.

    Parameters
    ----------
    data : ModelData
        Data for the model
    naive : bool, default False
        Whether the model uses simple floor rounding in likelihood

    Returns
    -------
    pymc.Model
        Throw time model

    See Also
    --------
    The structure of the model, the error model and the motivation for it is explained
    more thoroughly in the report stored in KyykkaAnalysis repo.
    """

    coordinates = {
        "players": data.player_names,
        "throws": np.arange(data.throw_times.size),
    }
    model = pm.Model(coords=coordinates)
    with model:
        mu = pm.TruncatedNormal("mu", mu=28, sigma=11, lower=0)
        sigma = pm.HalfNormal("sigma", sigma=11)
        o = pm.HalfNormal("o", sigma=11)
        k = pm.TruncatedNormal("k", mu=1, sigma=14, lower=1)

        theta = pm.TruncatedNormal("theta", mu=mu, sigma=sigma, lower=0, dims="players")
        player = pm.Data("player", data.player_ids, dims="throws")
        is_first = pm.Data("is_first", data.first_throw, dims="throws")
        throw_times = pm.Data("throw_times", data.throw_times, dims="throws")

        if not naive:
            pm.CustomDist(
                "y",
                k,
                theta[player] + o * is_first,
                logp=_podium_gamma_logp,
                random=_podium_gamma_rng,
                dims="throws",
                observed=throw_times,
            )
        else:
            pm.CustomDist(
                "y",
                k,
                theta[player] + o * is_first,
                dist=_floored_gamma,
                dims="throws",
                observed=throw_times,
            )

    return model


def _gammainc(k: float | TensorVariable, x: DATA_INPUT_TYPE) -> TensorVariable:
    return switch(lt(x, 0), 0, gammainc(k, x))


def _gammaincc(k: float | TensorVariable, x: DATA_INPUT_TYPE) -> TensorVariable:
    return switch(lt(x, 0), 0, gammaincc(k, x))


def _podium_gamma_logp(
    value: DATA_INPUT_TYPE,
    k: float | TensorVariable,
    theta: float | TensorVariable,
) -> TensorVariable:
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

    # Differences of gamma distributions CDF
    densities = densities[3:, :][::-1, :] - densities[:3, :]
    weights = expand_dims([5 / 9, 3 / 9, 1 / 9], -1)

    return log(sum(densities * weights, 0))


def _floored_gamma(
    k: float | TensorVariable, theta: float | TensorVariable, size: Shape
) -> TensorVariable:
    return floor(pm.Gamma.dist(alpha=k, beta=k / theta, size=size))


def invgamma_throw_model(data: ModelData, *, naive: bool = False) -> pm.Model:
    """
    Construct a model for throw times.

    Constructs a model for throw times which uses inverse gamma distribution for the
    underlying distribution of noise free throw times. The error model is either floor
    rounding or a more complex distribution based on error distribution of timestamps.

    Parameters
    ----------
    data : ModelData
        Data for the model
    naive : bool, default False
        Whether the model uses simple floor rounding in likelihood

    Returns
    -------
    pymc.Model
        Throw time model

    See Also
    --------
    The structure of the model and the motivation for it is explained more thoroughly in
    the report stored in kyykkaanalysis repo.
    """

    coordinates = {
        "players": data.player_names,
        "throws": np.arange(data.throw_times.size),
    }
    model = pm.Model(coords=coordinates)
    with model:
        mu = pm.TruncatedNormal("mu", mu=28, sigma=11, lower=0)
        sigma = pm.HalfNormal("sigma", sigma=11)
        o = pm.HalfNormal("o", sigma=11)
        a = pm.Normal("a", mu=-4.5, sigma=1)

        theta = pm.TruncatedNormal("theta", mu=mu, sigma=sigma, lower=0, dims="players")
        player = pm.Data("player", data.player_ids, dims="throws")
        is_first = pm.Data("is_first", data.first_throw, dims="throws")
        throw_times = pm.Data("throw_times", data.throw_times, dims="throws")

        if not naive:
            pm.CustomDist(
                "y",
                a,
                theta[player] + o * is_first,
                logp=_podium_invgamma_logp,
                random=_podium_invgamma_rng,
                dims="throws",
                observed=throw_times,
            )
        else:
            pm.CustomDist(
                "y",
                a,
                theta[player] + o * is_first,
                dist=_floored_invgamma,
                dims="throws",
                observed=throw_times,
            )

    return model


def _podium_invgamma_logp(
    value: npt.ArrayLike | Sequence[TensorVariable],
    a: float | TensorVariable,
    theta: float | TensorVariable,
) -> float | TensorVariable:
    alpha = exp(-a) + 1
    beta = theta * exp(-a)

    density = switch(
        value > beta / (alpha - 1 + 1e-9),
        _gammainc_diff(value, alpha, beta),
        _invdist_diff(value, alpha, beta),
    )

    return log(5 / 9 * density[0] + 3 / 9 * density[1] + 1 / 9 * density[2])


def _invdist_diff(
    value: npt.ArrayLike | Sequence[TensorVariable],
    alpha: float | TensorVariable,
    beta: float | TensorVariable,
) -> tuple[float] | Sequence[TensorVariable]:
    dist = pm.InverseGamma.dist(alpha=alpha, beta=beta)

    density1 = exp(pm.logcdf(dist, value + 3)) - exp(pm.logcdf(dist, value - 2))
    density2 = exp(pm.logcdf(dist, value + 2)) - exp(pm.logcdf(dist, value - 1))
    density3 = exp(pm.logcdf(dist, value + 1)) - exp(pm.logcdf(dist, value))

    return density1, density2, density3


def _gammainc_diff(
    value: npt.ArrayLike | Sequence[TensorVariable],
    alpha: float | TensorVariable,
    beta: float | TensorVariable,
) -> tuple[float] | Sequence[TensorVariable]:
    density1 = gammainc(alpha, beta / (value - 2)) - gammainc(alpha, beta / (value + 3))
    density2 = gammainc(alpha, beta / (value - 1)) - gammainc(alpha, beta / (value + 2))
    density3 = gammainc(alpha, beta / (value)) - gammainc(alpha, beta / (value + 1))

    return density1, density2, density3


def _podium_invgamma_rng(
    a: float,
    theta: float,
    *,
    rng: np.random.RandomState | np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> npt.NDArray[np.int_]:
    if rng is None:
        rng = np.random.default_rng()

    # If x ~ gamma(alpha,beta) -> 1/x ~ inv-gamma(alpha,beta)
    draws = 1 / rng.gamma(np.exp(-a) + 1, np.exp(a) / theta, size=size)
    draws += (
        rng.multinomial(1, [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9], size=size).argmax(1) - 2
    )

    return np.floor(draws)


def _floored_invgamma(
    a: float | TensorVariable, theta: float | TensorVariable, size: Shape
) -> TensorVariable:
    return floor(
        pm.InverseGamma.dist(alpha=exp(-a) + 1, beta=theta * exp(-a), size=size)
    )
