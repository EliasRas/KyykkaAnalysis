"""Checking priors and model definition"""

from pathlib import Path

import numpy as np
from xarray import Dataset, open_dataset
from scipy.stats import kstest

from .modeling import ThrowTimeModel
from ..data.data_classes import Stream, ModelData
from ..figures.prior import parameter_distributions as prior_distribution_plots
from ..figures.posterior import (
    parameter_distributions as posterior_distribution_plots,
    predictive_distributions,
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

    naive_figure_directory = figure_directory / "naive_SimulatedData"
    figure_directory = figure_directory / "SimulatedData"
    naive_cache_directory = cache_directory / "naive"
    naive_cache_directory.mkdir(parents=True, exist_ok=True)

    _test_model(data, figure_directory, cache_directory)
    _test_naive_model(data, naive_figure_directory, naive_cache_directory)


def _test_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    model = ThrowTimeModel(ModelData(data))
    naive_model = ThrowTimeModel(ModelData(data), naive=True)

    prior_file = cache_directory / "prior.nc"
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent, cache_directory)
        prior = open_dataset(prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        naive_model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _test_naive_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    model = ThrowTimeModel(ModelData(data))
    naive_model = ThrowTimeModel(ModelData(data), naive=True)

    naive_prior_file = cache_directory / "naive_prior.nc"
    if naive_prior_file.exists():
        naive_prior = open_dataset(naive_prior_file)
    else:
        check_priors(data, figure_directory.parent, cache_directory, naive=True)
        naive_prior = open_dataset(naive_prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        model, naive_prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        naive_model, naive_prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _fake_data_inference(
    model: ThrowTimeModel,
    prior: Dataset,
    cache_directory: Path,
    figure_directory: Path,
) -> tuple[Dataset, Dataset]:
    parameters = sorted(set(prior.keys()) - model.observed_variables)
    sample_count = min(100, len(prior.coords["draw"]))
    summaries, predictive_summaries = _generate_summaries(
        parameters, sample_count, prior, model
    )

    for sample_index in range(sample_count):
        sample = prior.isel(draw=sample_index)
        for parameter in parameters:
            true_value = sample[parameter]
            if "players" not in true_value.coords:
                summaries[parameter].sel(summary="true value")[sample_index] = (
                    true_value.values.item()
                )
                summaries[parameter].sel(summary="prior std")[sample_index] = prior[
                    parameter
                ].values.std()
            else:
                summaries[parameter].sel(summary="true value")[sample_index] = (
                    true_value.values.squeeze()
                )
                summaries[parameter].sel(summary="prior std")[sample_index] = (
                    prior[parameter].values.squeeze().std(0)
                )

        posterior_sample = _sample_posterior(
            sample_index, sample, cache_directory, model
        )
        thinned_sample = model.thin(posterior_sample)
        posterior_predictive = model.sample_posterior_predictive(thinned_sample)

        if sample_index in [0, 1, 4]:
            _visualize_sample(
                model,
                figure_directory,
                sample_index,
                prior,
                sample,
                posterior_sample,
                posterior_predictive,
            )

        for parameter in parameters:
            sample = thinned_sample[parameter]
            summaries[parameter].sel(summary="sample size").values[sample_index] = (
                sample.shape[0] * sample.shape[1]
            )

        _summarize_posterior(summaries, posterior_sample, sample_index)
        _summarize_posterior_predictive(
            predictive_summaries,
            posterior_predictive,
            prior[posterior_predictive.keys()].isel(draw=sample_index),
            sample_index,
        )

    return summaries, predictive_summaries


def _generate_summaries(
    parameters: list[str], sample_count: int, prior: Dataset, model: ThrowTimeModel
) -> tuple[Dataset, Dataset]:
    summary_statistics = [
        "true value",
        "conditional mean",
        "posterior std",
        "percentile",
        "sample size",
        "prior std",
    ]

    data_vars = {}
    for parameter in parameters:
        if "players" in prior[parameter].coords:
            data_vars[parameter] = (
                ["draw", "summary", "players"],
                np.zeros(
                    (sample_count, len(summary_statistics), prior[parameter].shape[-1])
                ),
            )
        else:
            data_vars[parameter] = (
                ["draw", "summary"],
                np.zeros((sample_count, len(summary_statistics))),
            )
    summaries = Dataset(
        data_vars=data_vars,
        coords={
            "draw": np.arange(sample_count),
            "players": prior.coords["players"],
            "summary": summary_statistics,
        },
    )
    predictive_summaries = Dataset(
        {
            var: (["draw", "summary"], np.zeros((sample_count, 1)))
            for var in model.observed_variables
        },
        coords={"draw": np.arange(sample_count), "summary": ["KS distance"]},
    )

    return summaries, predictive_summaries


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


def _visualize_sample(
    model: ThrowTimeModel,
    figure_directory: Path,
    sample_index: int,
    prior: Dataset,
    sample: Dataset,
    posterior_sample: Dataset,
    posterior_predictive: Dataset,
) -> None:
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
    predictive_distributions(
        posterior_predictive,
        sample_directory,
        data=prior[posterior_predictive.keys()].isel(draw=sample_index),
    )
    chain_plots(posterior_sample, sample_directory)


def _summarize_posterior(
    summaries: Dataset,
    sample: Dataset,
    sample_index: int,
) -> None:
    for parameter in summaries.keys():
        parameter_sample = sample[parameter]
        true_value = summaries[parameter].sel(summary="true value")[sample_index].values

        if "players" not in parameter_sample.coords:
            parameter_sample = parameter_sample.values

            summaries[parameter].sel(summary="conditional mean")[sample_index] = (
                parameter_sample.mean()
            )
            summaries[parameter].sel(summary="posterior std")[sample_index] = (
                parameter_sample.std()
            )
            summaries[parameter].sel(summary="percentile")[sample_index] = (
                parameter_sample < true_value
            ).sum() / parameter_sample.size
        else:
            parameter_sample = parameter_sample.values

            summaries[parameter].sel(summary="conditional mean")[sample_index] = (
                parameter_sample.mean((0, 1))
            )
            summaries[parameter].sel(summary="posterior std")[sample_index] = (
                parameter_sample.std((0, 1))
            )
            summaries[parameter].sel(summary="percentile")[sample_index] = (
                parameter_sample < true_value
            ).sum((0, 1)) / (parameter_sample.shape[0] * parameter_sample.shape[1])


def _summarize_posterior_predictive(
    summaries: Dataset,
    sample: Dataset,
    prior_sample: Dataset,
    sample_index: int,
) -> None:
    for parameter in summaries.keys():
        parameter_sample = sample[parameter].values.flatten()
        prior_parameter_sample = prior_sample[parameter].values.flatten()
        summaries[parameter].sel(summary="KS distance")[sample_index] = kstest(
            parameter_sample, prior_parameter_sample
        ).statistic
