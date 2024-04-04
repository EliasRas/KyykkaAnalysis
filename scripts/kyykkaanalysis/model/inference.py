"""Fitting the model"""

from pathlib import Path

from xarray import Dataset, open_dataset
from arviz import summary, InferenceData

from .modeling import ThrowTimeModel
from .model_checks import check_priors
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

    prior_file = cache_directory / "prior.nc"
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent, cache_directory)
        prior = open_dataset(prior_file)

    model = ThrowTimeModel(ModelData(data))
    posterior = _sample_posterior(model, cache_directory)
    _visualize_sample(
        prior,
        model.dataset,
        posterior,
        thinned_posterior,
        posterior_predictive,
        figure_directory,
    )

    naive_model = ThrowTimeModel(ModelData(data), naive=True)
    naive_posterior = _sample_posterior(naive_model, cache_directory)
    _visualize_sample(
        prior,
        naive_model.dataset,
        naive_posterior,
        naive_thinned_posterior,
        naive_posterior_predictive,
        naive_figure_directory,
    )


def _sample_posterior(model: ThrowTimeModel, cache_directory: Path) -> InferenceData:
    if model.naive:
        posterior_file = cache_directory / "naive_posterior.nc"
    else:
        posterior_file = cache_directory / "posterior.nc"

    if posterior_file.exists():
        posterior = InferenceData.from_netcdf(posterior_file)
    else:
        posterior = model.sample(
            sample_count=10000, chain_count=4, parallel_count=4, thin=False
        )
        posterior.to_netcdf(posterior_file)

    posterior.thinned_posterior = model.thin_posterior(posterior.posterior)
    posterior.posterior_predictive = model.sample_posterior_predictive(
        posterior.thinned_posterior
    )

    return posterior


def _visualize_sample(
    prior: Dataset,
    data: Dataset,
    posterior: InferenceData,
    figure_directory: Path,
) -> None:
    raw_directory = figure_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    parameter_distributions(posterior.posterior, raw_directory, prior)
    chain_plots(posterior.posterior, posterior.sample_stats, raw_directory)

    parameter_distributions(posterior.thinned_posterior, figure_directory, prior)
    chain_plots(posterior.thinned_posterior, posterior.sample_stats, figure_directory)

    predictive_distributions(posterior.posterior_predictive, figure_directory, data)
