"""Checking priors and model definition"""

from pathlib import Path

import numpy as np
from xarray import Dataset, open_dataset

from .modeling import ThrowTimeModel
from ..data.data_classes import Stream, ModelData
from ..figures.prior import parameter_distributions as prior_distribution_plots
from ..figures.posterior import (
    parameter_distributions as posterior_distribution_plots,
    chain_plots,
)
from ..figures.fake_data import estimation_plots


def check_priors(
    data: list[Stream],
    figure_directory: Path,
    cache_directory: Path,
    naive: bool = False,
):
    """
    Sample data from prior distribution and analyze it

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    cache_directory : Path
        Path to the directory in which the sampled prior is saved
    naive : bool, default False
        Whether to use simple floor rounding in likelihood
    """

    figure_directory = figure_directory / "Prior"
    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data), naive=naive)
    if naive:
        cache_file = cache_directory / "naive_prior.nc"
    else:
        cache_file = cache_directory / "prior.nc"
    if cache_file.exists():
        samples = open_dataset(cache_file)
    else:
        samples = model.sample_prior(10000)
        samples.to_netcdf(cache_file)

    prior_distribution_plots(
        samples, model.data.player_ids, model.data.first_throw, figure_directory
    )


def fake_data_simulation(
    data: list[Stream], figure_directory: Path, cache_directory: Path
):
    """
    Simulate data from prior and check if the parameter values can be recovered

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    cache_directory : Path
        Path to the directory in which the sampled prior is saved
    """

    figure_directory = figure_directory / "SimulatedData"
    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data))
    prior_file = cache_directory / "priori.nc"
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent, cache_directory)
        prior = open_dataset(prior_file)

    simulation_summaries = _fake_data_inference(
        model, prior, cache_directory, figure_directory
    )
    estimation_plots(simulation_summaries, figure_directory)


def _fake_data_inference(
    model: ThrowTimeModel,
    prior: Dataset,
    cache_directory: Path,
    figure_directory: Path,
) -> Dataset:
    parameters = sorted(set(prior.keys()) - {"y"})
    sample_count = min(100, len(prior.coords["draw"]))
    data_vars = {}
    for parameter in parameters:
        if "players" in prior[parameter].coords:
            data_vars[parameter] = (
                ["draw", "summary", "players"],
                np.zeros((sample_count, 5, prior[parameter].shape[-1])),
            )
        else:
            data_vars[parameter] = (
                ["draw", "summary"],
                np.zeros((sample_count, 5)),
            )
    posterior_summaries = Dataset(
        data_vars=data_vars,
        coords={
            "draw": np.arange(sample_count),
            "players": prior.coords["players"],
            "summary": [
                "true value",
                "conditional mean",
                "posterior std",
                "percentile",
                "sample size",
            ],
        },
    )

    for sample_index in range(sample_count):
        sample = prior.isel(draw=sample_index)
        for parameter in parameters:
            true_value = sample[parameter]
            if "players" not in true_value.coords:
                posterior_summaries[parameter].sel(summary="true value")[
                    sample_index
                ] = true_value.values.item()
            else:
                posterior_summaries[parameter].sel(summary="true value")[
                    sample_index
                ] = true_value.values.squeeze()

        posterior_sample = _sample_posterior(
            sample_index, sample, cache_directory, model
        )

        if sample_index in [0, 1, 4]:
            if model.naive:
                sample_directory = figure_directory / f"naive_{sample_index}"
            else:
                sample_directory = figure_directory / str(sample_index)
            posterior_distribution_plots(
                posterior_sample,
                sample_directory,
                prior_samples=prior,
                true_values=sample,
            )
            chain_plots(posterior_sample, sample_directory)

        posterior_sample = model.thin(posterior_sample)
        for parameter in parameters:
            sample = posterior_sample[parameter]
            posterior_summaries[parameter].sel(summary="sample size").values[
                sample_index
            ] = (
                sample.shape[0] * sample.shape[1]  # pylint: disable=superfluous-parens
            )

        _summarize_posterior(posterior_summaries, posterior_sample, sample_index)

    return posterior_summaries


def _sample_posterior(
    sample_index: int, sample: Dataset, cache_directory: Path, model: ThrowTimeModel
) -> Dataset:
    if model.naive:
        posterior_file = cache_directory / f"naive_posterior_{sample_index}.nc"
    else:
        posterior_file = cache_directory / f"posterior_{sample_index}.nc"
    if posterior_file.exists():
        posterior_sample = open_dataset(posterior_file)
    else:
        model.change_observations(y=sample["y"].values)
        posterior_sample = model.sample(
            sample_count=1000, chain_count=4, parallel_count=4, thin=False
        )
        posterior_sample.to_netcdf(posterior_file)

    return posterior_sample


def _summarize_posterior(
    posterior_summaries: Dataset,
    posterior_sample: Dataset,
    sample_index: int,
) -> None:
    for parameter in posterior_summaries.keys():
        parameter_sample = posterior_sample[parameter]
        true_value = (
            posterior_summaries[parameter]
            .sel(summary="true value")[sample_index]
            .values
        )

        if "players" not in parameter_sample.coords:
            parameter_sample = parameter_sample.values

            posterior_summaries[parameter].sel(summary="conditional mean")[
                sample_index
            ] = parameter_sample.mean()
            posterior_summaries[parameter].sel(summary="posterior std")[
                sample_index
            ] = parameter_sample.std()
            posterior_summaries[parameter].sel(summary="percentile")[sample_index] = (
                parameter_sample < true_value
            ).sum() / parameter_sample.size
        else:
            parameter_sample = parameter_sample.values

            posterior_summaries[parameter].sel(summary="conditional mean")[
                sample_index
            ] = parameter_sample.mean((0, 1))
            posterior_summaries[parameter].sel(summary="posterior std")[
                sample_index
            ] = parameter_sample.std((0, 1))
            posterior_summaries[parameter].sel(summary="percentile")[sample_index] = (
                parameter_sample < true_value
            ).sum((0, 1)) / (parameter_sample.shape[0] * parameter_sample.shape[1])
