"""Construct a playtime model"""

from typing import Self

import numpy as np
from numpy import typing as npt
from xarray import merge
from xarray import Dataset, DataArray
import pymc as pm
from pymc.math import floor, exp, log
from arviz import summary, loo, loo_pit, psislw, ess, InferenceData, ELPDData

from ..data.data_classes import ModelData


class ThrowTimeModel:
    """
    Container for throw time model

    Attributes
    ----------
    data : ModelData
        Data for the model
    naive : bool
        Whether the model uses simple floor rounding in likelihood
    model : pymc.Model
        Throw time model
    """

    # Should be changed to https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html

    def __init__(self, data: ModelData, naive: bool = False) -> None:
        """
        Container for throw time model

        Parameters
        ----------
        data : ModelData
            Data for the model
        """

        self.data = data
        self.naive = naive
        if not naive:
            self.model = gamma_throw_model(data)
        else:
            self.model = naive_throw_model(data)

    def sample_prior(self, sample_count: int = 500) -> Dataset:
        """
        Sample from the prior predictive distribution

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

    def sample(
        self,
        sample_count: int = 1000,
        tune_count: int = 1000,
        chain_count: int = 4,
        parallel_count: int = 1,
        thin: bool = True,
    ) -> InferenceData:
        """
        Sample from the posterior distribution

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
            starting_point = {
                "mu_interval__": np.array(np.log(28)),
                "sigma_log__": np.array(np.log(11)),
                "o_log__": np.array(np.log(1)),
                "k_interval__": np.array(np.log(2)),
                "theta_interval__": np.ones(len(self.model.coords["players"]))
                * np.log(28),
            }
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

    def sample_posterior_predictive(self, posterior_sample: Dataset) -> Dataset:
        """
        Sample from the posterior predictive distribution

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
        Carry out Pareto smoothed importance sampling leave-one-out cross-validation

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
            log_weights=self._psis_weights(
                posterior_samples, posterior.log_likelihood  # pylint: disable=no-member
            ),
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

    def change_observations(self, y: npt.NDArray[np.int_] | None = None) -> Self:
        """
        Change the observed data in the model

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

                # Ensure that measurement error bounds are within Gamma distribution domain
                # Otherwise gradient calculation fails
                if self.naive:
                    y[y < 1] = 1
                else:
                    y[y < 3] = 3

                pm.set_data({"throw_times": y})

        return self

    @staticmethod
    def thin(samples: InferenceData) -> InferenceData:
        """
        Thin the chains to reduce autocorrelations

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
        Thin the chains to reduce autocorrelations

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
        Names of the observed variables

        Returns
        -------
        set of str
            Observed variables
        """

        return {"y"}

    @property
    def dataset(self) -> Dataset:
        """
        Data for the model as xarray.Dataset

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


def gamma_throw_model(data: ModelData) -> pm.Model:
    """
    Construct a model for throw times

    Parameters
    ----------
    data : ModelData
        Data for the model

    Returns
    -------
    pymc.Model
        Throw time model
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
        player = pm.MutableData("player", data.player_ids, dims="throws")
        is_first = pm.MutableData("is_first", data.first_throw, dims="throws")
        throw_times = pm.MutableData("throw_times", data.throw_times, dims="throws")

        pm.CustomDist(
            "y",
            k,
            theta[player] + o * is_first,  # pylint: disable=unsubscriptable-object
            logp=_podium_gamma_logp,
            random=_podium_gamma_rng,
            dims="throws",
            observed=throw_times,
        )

    return model


def _podium_gamma_logp(value, k: float, theta: float):
    dist = pm.Gamma.dist(alpha=k, beta=k / theta)

    density1 = exp(pm.logcdf(dist, value + 3)) - exp(pm.logcdf(dist, value - 2))
    density2 = exp(pm.logcdf(dist, value + 2)) - exp(pm.logcdf(dist, value - 1))
    density3 = exp(pm.logcdf(dist, value + 1)) - exp(pm.logcdf(dist, value))

    return log(5 / 9 * density1 + 3 / 9 * density2 + 1 / 9 * density3)


def _podium_gamma_rng(
    k: float,
    theta: float,
    rng: np.random.RandomState | np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> npt.NDArray[np.int_]:
    if rng is None:
        rng = np.random.default_rng()

    draws = rng.gamma(k, theta / k, size=size)
    draws += (
        rng.multinomial(1, [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9], size=size).argmax(1) - 2
    )

    return np.floor(draws)


def naive_throw_model(data: ModelData) -> pm.Model:
    """
    Construct a model for throw times

    Parameters
    ----------
    data : ModelData
        Data for the model

    Returns
    -------
    pymc.Model
        Throw time model
    """

    coordinates = {
        "players": data.player_names,
        "throws": np.arange(data.throw_times.size),
    }
    model = pm.Model(coords=coordinates)
    with model:
        mu = pm.TruncatedNormal("mu", mu=28, sigma=11, lower=0, upper=np.inf)
        sigma = pm.HalfNormal("sigma", sigma=11)
        o = pm.HalfNormal("o", sigma=11)
        k = pm.TruncatedNormal("k", mu=1, sigma=14, lower=1)

        theta = pm.TruncatedNormal("theta", mu=mu, sigma=sigma, lower=0, dims="players")
        player = pm.MutableData("player", data.player_ids, dims="throws")
        is_first = pm.MutableData("is_first", data.first_throw, dims="throws")
        throw_times = pm.MutableData("throw_times", data.throw_times, dims="throws")

        pm.CustomDist(
            "y",
            k,
            theta[player] + o * is_first,  # pylint: disable=unsubscriptable-object
            dist=_floored_gamma,
            dims="throws",
            observed=throw_times,
        )

    return model


def _floored_gamma(k: float, theta: float, size: int):
    return floor(pm.Gamma.dist(alpha=k, beta=k / theta, size=size))
