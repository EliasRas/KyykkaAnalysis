"""
Constructing play time models.

This module constructs kyykkä play time models and provides containers for them with
functionalities for interacting with the models.
"""

from enum import Enum
from typing import Self

import numpy as np
import pymc as pm
from arviz import ELPDData, InferenceData, ess, loo, loo_pit, psislw, summary
from numpy import typing as npt
from xarray import DataArray, Dataset, merge

from ..data.data_classes import ModelData
from .distributions import (
    floor_gamma_logp,
    floor_gamma_rng,
    floor_invgamma_logp,
    floor_invgamma_rng,
    podium_gamma_logp,
    podium_gamma_rng,
    podium_invgamma_logp,
    podium_invgamma_rng,
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
        self,
        data: ModelData,
        *,
        model_type: ModelType = ModelType.GAMMA,
        non_centered: bool = False,
        extra_stability: bool = False,
    ) -> None:
        """
        Container for throw time model.

        Parameters
        ----------
        data : ModelData
            Data for the model
        model_type : ModelType, default GAMMA
            Type of model to construct
        non_centered : bool, default False
            Whether the model uses a non-centered parametrization
        extra_stability : bool, default False
            Whether the model uses a likelihood implementation that's numerically more
            stable
        """

        self.data = data
        self.model_type = model_type
        self._non_centered = non_centered
        self._extra_stability = extra_stability
        self._create_model()

    def _create_model(self) -> None:
        match self.model_type:
            case ModelType.GAMMA:
                self.model = gamma_throw_model(
                    self.data,
                    non_centered=self._non_centered,
                    extra_stability=self._extra_stability,
                )
            case ModelType.NAIVE:
                self.model = gamma_throw_model(
                    self.data,
                    naive=True,
                    non_centered=self._non_centered,
                    extra_stability=self._extra_stability,
                )
            case ModelType.INVGAMMA:
                self.model = invgamma_throw_model(
                    self.data,
                    non_centered=self._non_centered,
                    extra_stability=self._extra_stability,
                )
            case ModelType.NAIVEINVGAMMA:
                self.model = invgamma_throw_model(
                    self.data,
                    naive=True,
                    non_centered=self._non_centered,
                    extra_stability=self._extra_stability,
                )

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

            if self._non_centered:
                # Transform non-centered parametrization back to centered
                samples.posterior["theta"] = (
                    samples.posterior["mu"]
                    + samples.posterior["sigma"] * samples.posterior["eta"]
                )

        if thin:
            return self.thin(samples)

        return samples

    def _starting_point(self) -> dict[str, npt.NDArray[np.float64]]:
        starting_point = {
            "mu_interval__": np.array(np.log(28)),
            "sigma_log__": np.array(np.log(11)),
            "o_log__": np.array(np.log(1)),
        }

        if self._non_centered:
            starting_point["eta"] = np.zeros(len(self.model.coords["players"]))
        else:
            starting_point["theta_interval__"] = np.ones(
                len(self.model.coords["players"])
            ) * np.log(28)

        match self.model_type:
            case ModelType.GAMMA | ModelType.NAIVE:
                # k - 1 = 2
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

    def stability_check(
        self,
        max_value: float,
        theta: float,
        o: float,
        *,
        k: float | None = None,
        a: float | None = None,
    ) -> bool:
        """
        Check whether the log-likelihood calculation has to be numerically more stable.

        Checks whether the output of log-likelihood function is NaN for the given test
        values or the initial point given by _starting_point.

        Parameters
        ----------
        max_value : float
            Maximum throw time value
        theta : float
            Mean of the throw time distribution to test
        o : float
            Offset for the first throw
        k : float, optional
            Shape parameter of the gamma distribution to test. Used only if model_type
            is GAMMA or NAIVE
        a : float, optional
            Transformed shape parameter of inverse gamma distribution to test. Shape
            parameter alpha = exp(-a) + 1. Used only if model_type is INVGAMMA or
            NAIVEINVGAMMA

        Returns
        -------
        bool
            Whether the log-likelihood calculation has to be numerically more stable.
        """

        match self.model_type:
            case ModelType.GAMMA:
                logp = podium_gamma_logp(extra_stability=False)
                test_args = (max_value, k, theta + o)
                starting_args = (max_value, 3, 29)
            case ModelType.NAIVE:
                logp = floor_gamma_logp(extra_stability=False)
                test_args = (max_value, k, theta + o)
                starting_args = (max_value, 3, 29)
            case ModelType.INVGAMMA:
                logp = podium_invgamma_logp(extra_stability=False)
                test_args = (max_value, a, theta + o)
                starting_args = (max_value, -2, 29)
            case ModelType.NAIVEINVGAMMA:
                logp = floor_invgamma_logp(extra_stability=False)
                test_args = (max_value, a, theta + o)
                starting_args = (max_value, -2, 29)

        return not np.isfinite(logp(*test_args).eval().item()) or not np.isfinite(
            logp(*starting_args).eval().item()
        )

    def change_implementation(
        self,
        *,
        non_centered: bool | None = None,
        extra_stability: bool | None = None,
    ) -> Self:
        """
        Change the implementation of the model.

        Changes the parametrization of the model between centered and non-centered
        parameterizations and the implementation of the likelihood function between
        stable and fast.

        Parameters
        ----------
        non_centered : bool | None, optional
            Whether the model uses a non-centered parametrization
        extra_stability : bool | None, optional
            Whether the model uses a likelihood implementation that's numerically more
            stable

        Returns
        -------
        ThrowTimeModel
            Model with updated implementation
        """

        if non_centered is not None:
            self._non_centered = non_centered
        if extra_stability is not None:
            self._extra_stability = extra_stability

        self._create_model()

        return self

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


def gamma_throw_model(
    data: ModelData,
    *,
    naive: bool = False,
    non_centered: bool = False,
    extra_stability: bool = False,
) -> pm.Model:
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
    non_centered : bool, default False
        Whether the model uses non-centered parametrization
    extra_stability : bool, default False
        Whether the model uses a likelihood implementation that's numerically more
        stable

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

        if not non_centered:
            theta = pm.TruncatedNormal(
                "theta", mu=mu, sigma=sigma, lower=0, dims="players"
            )
        else:
            # Missing a lower bound -mu/sigma
            eta = pm.Normal("eta", mu=0, sigma=1, dims="players")
            theta = mu + sigma * eta
        player = pm.Data("player", data.player_ids, dims="throws")
        is_first = pm.Data("is_first", data.first_throw, dims="throws")
        throw_times = pm.Data("throw_times", data.throw_times, dims="throws")

        if not naive:
            pm.CustomDist(
                "y",
                k,
                theta[player] + o * is_first,
                logp=podium_gamma_logp(extra_stability=extra_stability),
                random=podium_gamma_rng,
                dims="throws",
                observed=throw_times,
            )
        else:
            pm.CustomDist(
                "y",
                k,
                theta[player] + o * is_first,
                logp=floor_gamma_logp(extra_stability=extra_stability),
                random=floor_gamma_rng,
                dims="throws",
                observed=throw_times,
            )

    return model


def invgamma_throw_model(
    data: ModelData,
    *,
    naive: bool = False,
    non_centered: bool = False,
    extra_stability: bool = False,
) -> pm.Model:
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
    non_centered : bool, default False
        Whether the model uses non-centered parametrization
    extra_stability : bool, default False
        Whether the model uses a likelihood implementation that's numerically more
        stable

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

        if not non_centered:
            theta = pm.TruncatedNormal(
                "theta", mu=mu, sigma=sigma, lower=0, dims="players"
            )
        else:
            eta = pm.Normal("eta", mu=0, sigma=1, dims="players")  # lower = -mu
            theta = mu + sigma * eta
        player = pm.Data("player", data.player_ids, dims="throws")
        is_first = pm.Data("is_first", data.first_throw, dims="throws")
        throw_times = pm.Data("throw_times", data.throw_times, dims="throws")

        if not naive:
            pm.CustomDist(
                "y",
                a,
                theta[player] + o * is_first,
                logp=podium_invgamma_logp(extra_stability=extra_stability),
                random=podium_invgamma_rng,
                dims="throws",
                observed=throw_times,
            )
        else:
            pm.CustomDist(
                "y",
                a,
                theta[player] + o * is_first,
                logp=floor_invgamma_logp(extra_stability=extra_stability),
                random=floor_invgamma_rng,
                dims="throws",
                observed=throw_times,
            )

    return model
