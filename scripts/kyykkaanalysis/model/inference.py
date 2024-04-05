"""Fitting the model"""

from pathlib import Path

from xarray import Dataset, open_dataset
from arviz import InferenceData

from .modeling import ThrowTimeModel
from .model_checks import check_priors
from ..data.data_classes import ModelData, Stream
from ..figures.posterior import (
    parameter_distributions,
    predictive_distributions,
)
from ..figures.chains import chain_plots
from ..figures.cross_validation import cross_validation_plots, model_comparison


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
        figure_directory,
    )

    naive_model = ThrowTimeModel(ModelData(data), naive=True)
    naive_posterior = _sample_posterior(naive_model, cache_directory)
    _visualize_sample(
        prior,
        naive_model.dataset,
        naive_posterior,
        naive_figure_directory,
    )

    _compare_models(
        model.dataset,
        {"Palkintopallivirhe": posterior, "Alaspäin pyöristys": naive_posterior},
        figure_directory,
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
    posterior.loo_result = model.psis_loo(
        posterior.thinned_posterior, posterior.posterior_predictive
    )

    return posterior


def _visualize_sample(
    prior: Dataset,
    data: Dataset,
    posterior: InferenceData,
    figure_directory: Path,
) -> None:
    raw_directory = figure_directory / "raw"
    parameter_distributions(posterior.posterior, raw_directory / "parameters", prior)
    chain_plots(posterior.posterior, posterior.sample_stats, raw_directory / "chains")

    parameter_distributions(
        posterior.thinned_posterior, figure_directory / "parameters", prior
    )
    chain_plots(
        posterior.thinned_posterior, posterior.sample_stats, figure_directory / "chains"
    )

    predictive_distributions(
        posterior.posterior_predictive, figure_directory / "predictions", data
    )
    cross_validation_plots(
        data, posterior.loo_result, figure_directory / "cross_validation"
    )


def _compare_models(
    data: Dataset, posteriors: dict[str, InferenceData], figure_directory: Path
) -> None:
    model_comparison(
        data,
        {
            model_name: posterior.loo_result
            for model_name, posterior in posteriors.items()
        },
        figure_directory / "cross_validation",
    )
