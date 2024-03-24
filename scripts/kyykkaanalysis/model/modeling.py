"""Construct a playtime model"""

from typing import Self

import numpy as np
from numpy import typing as npt
from xarray import merge
from xarray import Dataset
import pymc as pm
from pymc.math import floor, exp, log
from arviz import summary

from ..data.data_classes import ModelData


class ThrowTimeModel:
    """
    Container for throw time model

    Attributes
    ----------
    data : ModelData
        Data for the model
    naive : bool
        Whether the model uses only floor rounding in likelihood
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
            self.model = throw_model(data)
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
    ) -> Dataset:
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
        xarray.Dataset
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
            return self.thin(samples.posterior)

        return samples.posterior

    @staticmethod
    def thin(samples: Dataset) -> Dataset:
        """
        Thin the chains to reduce autocorrelations

        Parameters
        ----------
        samples : Dataset
            Samples from the chains

        Returns
        -------
        Dataset
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
                pm.set_data({"throw_times": y.flatten()})

        return self


def throw_model(data: ModelData) -> pm.Model:
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
