"""Construct a playtime model"""
from typing import Self

import numpy as np
from numpy import typing as npt
from scipy.stats import uniform
import pymc as pm
from xarray import merge
from xarray import Dataset

from ..data.data_classes import ModelData


class ThrowTimeModel:
    """
    Container for throw time model

    Attributes
    ----------
    data : ModelData
        Data for the model
    model : pymc.Model
        Throw time model
    """

    def __init__(self, data: ModelData) -> None:
        """
        Container for throw time model

        Parameters
        ----------
        data : ModelData
            Data for the model
        """

        self.data = data
        self.model = throw_model(data)

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

        y_hat = samples.prior_predictive["y_hat"]
        errors = self._sample_errors(y_hat.size).reshape(y_hat.shape)
        y = np.floor(y_hat + errors)
        y.values[y.values < 0] = 0
        samples.prior_predictive["y"] = y.astype(int)

        return merge([samples.prior.drop_vars(["k_minus"]), samples.prior_predictive])

    def _sample_errors(self, count: int) -> npt.NDArray[np.float_]:
        samples = uniform(0, 1).rvs(count)
        identifiers = uniform(0, 1).rvs(count)
        samples[identifiers < 5 / 9] = -3 + 5 * samples[identifiers < 5 / 9]
        samples[(identifiers >= 5 / 9) & (identifiers < 8 / 9)] = (
            -2 + 3 * samples[(identifiers >= 5 / 9) & (identifiers < 8 / 9)]
        )
        samples[identifiers >= 8 / 9] *= -1

        return samples

    def sample(
        self,
        sample_count: int = 1000,
        tune_count: int = 1000,
        chain_count: int = 4,
        parallel_count: int = 1,
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
                "k_minus_log__": np.array(np.log(1)),
                "theta_interval__": np.ones(len(self.model.coords["players"]))
                * np.log(28),
            }
            samples = pm.sample(
                draws=sample_count,
                tune=tune_count,
                chains=chain_count,
                cores=parallel_count,
                initvals=starting_point,
                init="jitter+adapt_diag",  # Testaa muita
            )

        return samples.posterior

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
        mu = pm.TruncatedNormal("mu", mu=28, sigma=11, lower=0, upper=np.inf)
        sigma = pm.HalfNormal("sigma", sigma=11)
        o = pm.HalfNormal("o", sigma=11)
        k_minus = pm.HalfNormal("k_minus", sigma=14)
        k = pm.Deterministic("k", k_minus + 1)

        theta = pm.TruncatedNormal("theta", mu=mu, sigma=sigma, lower=0, dims="players")
        player = pm.MutableData("player", data.player_ids, dims="throws")
        is_first = pm.MutableData("is_first", data.first_throw, dims="throws")
        throw_times = pm.MutableData("throw_times", data.throw_times, dims="throws")

        pm.Gamma(
            "y_hat",
            k,
            k
            / (theta[player] + o * is_first),  # pylint: disable=unsubscriptable-object
            dims="throws",
            observed=throw_times,  # Use the rounded measurements directly, wrong but seems to be the only way that works
        )

    return model
