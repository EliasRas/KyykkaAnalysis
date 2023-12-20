"""Checking priors and model definition"""
from typing import Any
from pathlib import Path

import numpy as np
from numpy import typing as npt
from xarray import Dataset, open_dataset

from .modeling import ThrowTimeModel
from ..data.data_classes import Stream, ModelData
from ..figures.prior import parameter_distributions as prior_distribution_plots
from ..figures.posterior import (
    parameter_distributions as posterior_distribution_plots,
    chain_plots,
)
from ..figures.fake_data import estimation_plots


def check_priors(data: list[Stream], figure_directory: Path, cache_directory: Path):
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
    """

    figure_directory = figure_directory / "Prior"
    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data))
    cache_file = cache_directory / "priori.nc"
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
    Simulate data from prior and check that the parameter values can be recovered

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

    parameters = sorted(set(prior.keys()) - {"k_minus", "y", "y_hat"})
    true_values = {parameter: [] for parameter in parameters}
    conditional_means = {parameter: [] for parameter in parameters}
    posterior_stds = {parameter: [] for parameter in parameters}
    percentiles = {parameter: [] for parameter in parameters}
    for sample_index in range(len(prior.coords["draw"])):
        sample = prior.isel(draw=sample_index)
        for parameter in parameters:
            true_value = sample[parameter].values
            if true_value.size == 1:
                true_values[parameter].append(true_value.item())
            else:
                true_values[parameter].append(true_value.squeeze())

        posterior_sample = _sample_posterior(
            sample_index, sample, cache_directory, model
        )

        if sample_index in []:
            posterior_distribution_plots(
                posterior_sample,
                figure_directory / str(sample_index),
                prior_samples=prior,
                true_values=sample,
            )
            chain_plots(posterior_sample, figure_directory / str(sample_index))

        posterior_sample = model.thin(posterior_sample)
        _summarize_posterior(
            parameters,
            {parameter: true_values[parameter][-1] for parameter in parameters},
            posterior_sample,
            conditional_means,
            posterior_stds,
            percentiles,
        )

    simulation_summaries = _summaries_to_dataset(
        parameters, prior, true_values, conditional_means, posterior_stds, percentiles
    )
    estimation_plots(simulation_summaries, figure_directory)


def _sample_posterior(
    sample_index: int, sample: Dataset, cache_directory: Path, model: ThrowTimeModel
) -> Dataset:
    posterior_file = cache_directory / f"posterior_{sample_index}.nc"
    if posterior_file.exists():
        posterior_sample = open_dataset(posterior_file)
    else:
        model.change_observations(y=sample["y_hat"].values)
        posterior_sample = model.sample(
            sample_count=1000, chain_count=4, parallel_count=4, thin=False
        )
        posterior_sample.to_netcdf(posterior_file)

    return posterior_sample


def _summarize_posterior(
    parameters: list[str],
    true_values: dict[str, Any | npt.NDArray[Any]],
    posterior_sample: Dataset,
    conditional_means: dict[str, list[float | npt.NDArray[Any]]],
    posterior_stds: dict[str, list[float | npt.NDArray[Any]]],
    percentiles: dict[str, list[float | npt.NDArray[Any]]],
) -> None:
    for parameter in parameters:
        parameter_sample = posterior_sample[parameter].values
        if len(parameter_sample.shape) == 2:
            conditional_means[parameter].append(parameter_sample.mean())
            posterior_stds[parameter].append(parameter_sample.std())
            percentiles[parameter].append(
                (parameter_sample < true_values[parameter]).sum()
                / parameter_sample.size
            )
        else:
            conditional_means[parameter].append(parameter_sample.mean((0, 1)))
            posterior_stds[parameter].append(parameter_sample.std((0, 1)))
            percentiles[parameter].append(
                (parameter_sample < true_values[parameter]).sum((0, 1))
                / (parameter_sample.shape[0] * parameter_sample.shape[1])
            )


def _summaries_to_dataset(
    parameters: list[str],
    prior: Dataset,
    true_values: dict[str, list[Any | npt.NDArray[Any]]],
    conditional_means: dict[str, list[float | npt.NDArray[Any]]],
    posterior_stds: dict[str, list[float | npt.NDArray[Any]]],
    percentiles: dict[str, list[float | npt.NDArray[Any]]],
) -> Dataset:
    data_vars = {}
    for parameter, parameter_truths in true_values.items():
        if isinstance(parameter_truths[0], np.ndarray):
            data_vars[parameter] = (
                [
                    "draw",
                    "summary",
                    "players",
                ],
                np.hstack(
                    (
                        np.expand_dims(parameter_truths, 1),
                        np.expand_dims(conditional_means[parameter], 1),
                        np.expand_dims(posterior_stds[parameter], 1),
                        np.expand_dims(percentiles[parameter], 1),
                    )
                ),
            )
        else:
            data_vars[parameter] = (
                ["draw", "summary"],
                np.hstack(
                    (
                        np.array(parameter_truths).reshape(-1, 1),
                        np.array(conditional_means[parameter]).reshape(-1, 1),
                        np.array(posterior_stds[parameter]).reshape(-1, 1),
                        np.array(percentiles[parameter]).reshape(-1, 1),
                    )
                ),
            )
    simulation_summaries = Dataset(
        data_vars=data_vars,
        coords={
            "draw": np.arange(len(true_values[parameters[0]])),
            "players": prior.coords["players"],
            "summary": [
                "true value",
                "conditional mean",
                "posterior std",
                "percentile",
            ],
        },
    )

    return simulation_summaries
