"""
Fitting a play time model.

This module performs statistical inference with different kinds of kyykkä play time
models. Their posterior distributions are sampled given the data and the samples are
stored for peristence. Posterior predictive checks are performed using the posteriors
and the performance of the models is compared using PSIS-LOO cross-validation. The
posteriors, posterior predictive checks and model comparisons are visualized
automatically.
"""

from pathlib import Path

from arviz import InferenceData
from xarray import Dataset, open_dataset

from ..data.data_classes import ModelData, Stream
from ..figures.chains import chain_plots
from ..figures.cross_validation import cross_validation_plots, model_comparison
from ..figures.posterior import (
    parameter_distributions,
    predictive_distributions,
)
from .model_checks import check_priors
from .modeling import ModelType, ThrowTimeModel


def fit_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    """
    Perform statistical inference with different models.

    Samples from the posterior distributions of different throw time models, and
    analyzes the models' performance using posterior predictive checks and PSIS-LOO
    cross-validation. Compares the performance of the models. Visualizes the MCMC
    chains, posterior distributions and analyses.

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
    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data))
    posterior = _sample_posterior(model, cache_directory)
    _visualize_sample(
        _load_prior(data, ModelType.GAMMA, figure_directory.parent, cache_directory),
        model.dataset,
        posterior,
        figure_directory,
    )

    naive_model = ThrowTimeModel(ModelData(data), model_type=ModelType.NAIVE)
    naive_posterior = _sample_posterior(naive_model, cache_directory)
    _visualize_sample(
        _load_prior(data, ModelType.NAIVE, figure_directory.parent, cache_directory),
        naive_model.dataset,
        naive_posterior,
        figure_directory / "naive",
    )

    inv_model = ThrowTimeModel(ModelData(data), model_type=ModelType.INVGAMMA)
    inv_posterior = _sample_posterior(inv_model, cache_directory)
    _visualize_sample(
        _load_prior(data, ModelType.INVGAMMA, figure_directory.parent, cache_directory),
        inv_model.dataset,
        inv_posterior,
        figure_directory / "inv",
    )

    naive_inv_model = ThrowTimeModel(
        ModelData(data), model_type=ModelType.NAIVEINVGAMMA
    )
    naive_inv_posterior = _sample_posterior(naive_inv_model, cache_directory)
    _visualize_sample(
        _load_prior(
            data, ModelType.NAIVEINVGAMMA, figure_directory.parent, cache_directory
        ),
        naive_inv_model.dataset,
        naive_inv_posterior,
        figure_directory / "naive_inv",
    )

    _compare_models(
        model.dataset,
        {
            "Palkintopallivirhe": posterior,
            "Alaspäin pyöristys": naive_posterior,
            "Käänteinen gammajakauma ja pp.virhe": inv_posterior,
            "Käänteinen gammajakauma ja pyöristysvirhe": naive_inv_posterior,
        },
        figure_directory,
    )


def _load_prior(
    data: list[Stream],
    model_type: ModelType,
    figure_directory: Path,
    cache_directory: Path,
) -> Dataset:
    match model_type:
        case ModelType.GAMMA:
            prior_file = cache_directory / "prior.nc"
        case ModelType.NAIVE:
            prior_file = cache_directory / "naive_prior.nc"
        case ModelType.INVGAMMA:
            prior_file = cache_directory / "inv_prior.nc"
        case ModelType.NAIVEINVGAMMA:
            prior_file = cache_directory / "naive_inv_prior.nc"

    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory, cache_directory, model_type=model_type)
        prior = open_dataset(prior_file)

    return prior


def _sample_posterior(model: ThrowTimeModel, cache_directory: Path) -> InferenceData:
    match model.model_type:
        case ModelType.GAMMA:
            posterior_file = cache_directory / "posterior.nc"
        case ModelType.NAIVE:
            posterior_file = cache_directory / "naive_posterior.nc"
        case ModelType.INVGAMMA:
            posterior_file = cache_directory / "inv_posterior.nc"
        case ModelType.NAIVEINVGAMMA:
            posterior_file = cache_directory / "naive_inv_posterior.nc"

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
    parameter_distributions(
        posterior.posterior, raw_directory / "parameters", prior_samples=prior
    )
    chain_plots(
        posterior.posterior,
        raw_directory / "chains",
        sample_stats=posterior.sample_stats,
    )

    parameter_distributions(
        posterior.thinned_posterior,
        figure_directory / "parameters",
        prior_samples=prior,
    )
    chain_plots(
        posterior.thinned_posterior,
        figure_directory / "chains",
        sample_stats=posterior.sample_stats,
    )

    predictive_distributions(
        posterior.posterior_predictive,
        figure_directory / "predictions",
        true_values=data,
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
