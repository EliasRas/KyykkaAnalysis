"""Construct a playtime model"""
import numpy as np
import pymc as pm
from pymc.math import maximum, floor
from pymc.distributions.transforms import logodds
from xarray import Dataset, merge

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
        sample_count : int
            Number of samples to draw

        Returns
        -------
        xarray.Dataset
            Prior samples
        """

        with self.model:
            samples = pm.sample_prior_predictive(samples=sample_count)

        samples.prior["y"] = samples.prior["y"].astype(int)

        return samples.prior.drop_vars(["k_minus", "y_raw"])


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
        k_minus = pm.HalfNormal("k_minus", sigma=14)
        k = pm.Deterministic("k", k_minus + 1)

        theta = pm.TruncatedNormal("theta", mu=mu, sigma=sigma, lower=0, dims="players")
        player = pm.MutableData("player", data.player_ids, dims="throws")
        is_first = pm.MutableData("is_first", data.first_throw, dims="throws")
        y_hat = pm.Gamma(
            "y_hat",
            k,
            k
            / (theta[player] + o * is_first),  # pylint: disable=unsubscriptable-object
            dims="throws",
        )

        # Each timestamp have a uniform error distribution from -1 to 1
        # Hence the time between timestamps (y) has a winners' podium like distribution
        y_raw = pm.Mixture(
            "y_raw",
            w=[5 / 9, 1 / 3, 1 / 9],
            comp_dists=[
                pm.Uniform.dist(y_hat - 2, y_hat + 3),
                pm.Uniform.dist(y_hat - 1, y_hat + 2),
                pm.Uniform.dist(y_hat, y_hat + 1),
            ],
            dims="throws",
            observed=data.throw_times,
            transform=logodds,
        )
        # Manually discretize the times, faster than pm.DiscreteUniform
        pm.Deterministic("y", floor(maximum(y_raw, 0)), dims="throws")

    return model
