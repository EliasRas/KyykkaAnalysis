"""Fitting the model"""

from pathlib import Path

from xarray import Dataset, open_dataset

from .modeling import ThrowTimeModel
from ..data.data_classes import ModelData, Stream
from ..figures.posterior import (
    parameter_distributions,
    predictive_distributions,
    chain_plots,
)


def fit_model(data: list[Stream], figure_directory: Path, cache_directory: Path):
    """
    Simulate data from prior and check if the parameter values can be recovered

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    cache_directory : Path
        Path to the directory in which the sampled posterior is saved
    """

    figure_directory = figure_directory / "Posterior"
    naive_figure_directory = figure_directory / "naive"
    naive_figure_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data))
    posterior, thinned_posterior, posterior_predictive = _sample_posterior(
        model, cache_directory
    )
    _visualize_sample(
        model.dataset,
        posterior,
        thinned_posterior,
        posterior_predictive,
        figure_directory,
    )

    naive_model = ThrowTimeModel(ModelData(data), naive=True)
    naive_posterior, naive_thinned_posterior, naive_posterior_predictive = (
        _sample_posterior(naive_model, cache_directory)
    )
    _visualize_sample(
        model.dataset,
        naive_posterior,
        naive_thinned_posterior,
        naive_posterior_predictive,
        naive_figure_directory,
    )


def _sample_posterior(
    model: ThrowTimeModel, cache_directory: Path
) -> tuple[Dataset, Dataset, Dataset]:
    if model.naive:
        posterior_file = cache_directory / "naive_posterior.nc"
    else:
        posterior_file = cache_directory / "posterior.nc"

    if posterior_file.exists():
        posterior = open_dataset(posterior_file)
    else:
        posterior = model.sample(
            sample_count=1000, chain_count=4, parallel_count=4, thin=False
        )
        posterior.to_netcdf(posterior_file)

    thinned_posterior = model.thin(posterior)
    posterior_predictive = model.sample_posterior_predictive(posterior)

    return posterior, thinned_posterior, posterior_predictive


def _visualize_sample(
    data: Dataset,
    posterior: Dataset,
    thinned_posterior: Dataset,
    posterior_predictive: Dataset,
    figure_directory: Path,
) -> None:
    raw_directory = figure_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    parameter_distributions(posterior, raw_directory)
    chain_plots(posterior, raw_directory)

    parameter_distributions(thinned_posterior, figure_directory)
    chain_plots(thinned_posterior, figure_directory)

    predictive_distributions(posterior_predictive, figure_directory, data)
