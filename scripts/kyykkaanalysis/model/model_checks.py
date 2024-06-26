"""
Checking priors and parameter recovery.

This model performs and visualizes prior predictive checks and simulation based
calibration (SMC) for kyykkä play time models. SMC is performed with multiple models,
also with data generated with other models.
"""

from pathlib import Path
from typing import Any

import numpy as np
import structlog
from arviz import InferenceData, summary
from scipy.stats import kstest
from xarray import Dataset, open_dataset

from ..data.data_classes import ModelData, Stream
from ..figures.chains import chain_plots
from ..figures.fake_data import estimation_plots
from ..figures.posterior import (
    parameter_distributions as posterior_distribution_plots,
)
from ..figures.posterior import (
    predictive_distributions,
)
from ..figures.prior import parameter_distributions as prior_distribution_plots
from .modeling import ModelType, ThrowTimeModel

_LOG = structlog.get_logger(__name__)
_SBC_SAMPLES = 1000
_SBC_CHAINS = 4
_SMALL_SIGMA = 5


def check_priors(
    data: list[Stream],
    figure_directory: Path,
    cache_directory: Path,
    *,
    model_type: ModelType = ModelType.GAMMA,
) -> None:
    """
    Sample data from prior distribution and analyze it.

    Samples data from the prior distributions defined by the given model and visualizes
    the distributions of both the parameters and the observed variables.

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    cache_directory : Path
        Path to the directory in which the sampled prior is saved
    model_type : ModelType, default GAMMA
        Type of model to use
    """

    log = _LOG.bind(model_type=model_type)
    log.info("Performing prior predictive checks.")

    cache_directory.mkdir(parents=True, exist_ok=True)

    model = ThrowTimeModel(ModelData(data), model_type=model_type)
    match model_type:
        case ModelType.GAMMA:
            cache_file = cache_directory / "prior.nc"
        case ModelType.NAIVE:
            cache_file = cache_directory / "naive_prior.nc"
            figure_directory = figure_directory / "naive"
        case ModelType.INVGAMMA:
            cache_file = cache_directory / "inv_prior.nc"
            figure_directory = figure_directory / "inv"
        case ModelType.NAIVEINVGAMMA:
            cache_file = cache_directory / "naive_inv_prior.nc"
            figure_directory = figure_directory / "naiveinv"

    if cache_file.exists():
        samples = open_dataset(cache_file)
        log.debug("Prior samples read from cache.", cache_file=cache_file)
    else:
        samples = model.sample_prior(10000)
        samples.to_netcdf(cache_file)
        log.debug("Prior samples cached.", cache_file=cache_file)

    prior_distribution_plots(
        samples, model.data.player_ids, model.data.first_throw, figure_directory
    )

    log.info("Prior predictive checks completed.")


def fake_data_simulation(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    """
    Simulate data from prior and check if the parameter values can be recovered.

    Simulates data from the prior distributions defined by different kinds of models and
    samples from the posterior distribution given the prior predictive samples.
    Summarizes the parameter recovery and visualizes the results. Also visualizes the
    MCMC chains and the posterior distributions for the first chains and the chains
    where the sampling is not particularly efficient.

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    cache_directory : Path
        Path to the directory in which the sampled prior is saved
    """

    _LOG.info("Performing simulation based calibration.")

    naive_cache_directory = cache_directory / "naive"
    naive_cache_directory.mkdir(parents=True, exist_ok=True)

    _test_model(data, figure_directory / "SimulatedData", cache_directory)
    _test_naive_model(
        data, figure_directory / "naive_SimulatedData", naive_cache_directory
    )
    _test_inv_model(data, figure_directory / "inv_SimulatedData", cache_directory)
    _test_naive_inv_model(
        data, figure_directory / "naive_inv_SimulatedData", naive_cache_directory
    )

    _LOG.info("Simulation based calibration finished.")


def _test_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    log = _LOG.bind(data_model=ModelType.GAMMA)
    log.info("Performing simulation based calibration for a dataset.")

    model = ThrowTimeModel(ModelData(data))
    naive_model = ThrowTimeModel(ModelData(data), model_type=ModelType.NAIVE)

    prior_file = cache_directory / "prior.nc"
    log.debug("Loading prior samples.", cache_file=prior_file)
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent / "Prior", cache_directory)
        prior = open_dataset(prior_file)
    log.debug("Prior samples loaded.", cache_file=prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        log, model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        log, naive_model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _test_naive_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    log = _LOG.bind(data_model=ModelType.NAIVE)
    log.info("Performing simulation based calibration for a dataset.")

    model = ThrowTimeModel(ModelData(data))
    naive_model = ThrowTimeModel(ModelData(data), model_type=ModelType.NAIVE)

    prior_file = cache_directory / "naive_prior.nc"
    log.debug("Loading prior samples.", cache_file=prior_file)
    if prior_file.exists():
        naive_prior = open_dataset(prior_file)
    else:
        check_priors(
            data, figure_directory.parent, cache_directory, model_type=ModelType.NAIVE
        )
        naive_prior = open_dataset(prior_file)
    log.debug("Prior samples loaded.", cache_file=prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        log, model, naive_prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        log, naive_model, naive_prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _test_inv_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    log = _LOG.bind(data_model=ModelType.INVGAMMA)
    log.info("Performing simulation based calibration for a dataset.")

    model = ThrowTimeModel(ModelData(data), model_type=ModelType.INVGAMMA)
    naive_model = ThrowTimeModel(ModelData(data), model_type=ModelType.NAIVEINVGAMMA)

    prior_file = cache_directory / "inv_prior.nc"
    log.debug("Loading prior samples.", cache_file=prior_file)
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent / "Prior", cache_directory)
        prior = open_dataset(prior_file)
    log.debug("Prior samples loaded.", cache_file=prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        log, model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        log, naive_model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _test_naive_inv_model(
    data: list[Stream], figure_directory: Path, cache_directory: Path
) -> None:
    log = _LOG.bind(data_model=ModelType.INVGAMMA)
    log.info("Performing simulation based calibration for a dataset.")

    model = ThrowTimeModel(ModelData(data), model_type=ModelType.INVGAMMA)
    naive_model = ThrowTimeModel(ModelData(data), model_type=ModelType.NAIVEINVGAMMA)

    prior_file = cache_directory / "naive_inv_prior.nc"
    log.debug("Loading prior samples.", cache_file=prior_file)
    if prior_file.exists():
        prior = open_dataset(prior_file)
    else:
        check_priors(data, figure_directory.parent / "Prior", cache_directory)
        prior = open_dataset(prior_file)
    log.debug("Prior samples loaded.", cache_file=prior_file)

    summaries, predictive_summaries = _fake_data_inference(
        log, model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory)

    summaries, predictive_summaries = _fake_data_inference(
        log, naive_model, prior, cache_directory, figure_directory
    )
    estimation_plots(summaries, predictive_summaries, figure_directory / "naive")


def _fake_data_inference(
    log: Any,
    model: ThrowTimeModel,
    prior: Dataset,
    cache_directory: Path,
    figure_directory: Path,
) -> tuple[Dataset, Dataset]:
    log = log.bind(model_type=model.model_type)
    parameters = sorted(set(prior.keys()) - model.observed_variables)
    sample_count = min(100, len(prior.coords["draw"]))
    log = log.bind(sample_count=sample_count)
    summaries, predictive_summaries = _generate_summaries(
        parameters, sample_count, prior, model
    )

    for sample_index in range(sample_count):
        log = log.bind(sample_index=sample_index)
        log.info(
            "Performing posterior inference on data sampled from prior distribution.",
        )
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

        match model.model_type:
            case ModelType.GAMMA | ModelType.NAIVE:
                needs_stability = model.stability_check(
                    sample["y"].values.max(),
                    sample["theta"].values.min(),
                    sample["o"].item(),
                    k=sample["k"].item(),
                )
            case ModelType.INVGAMMA | ModelType.NAIVEINVGAMMA:
                needs_stability = model.stability_check(
                    sample["y"].values.max(),
                    sample["theta"].values.min(),
                    sample["o"].item(),
                    a=sample["a"].item(),
                )
        if needs_stability or sample["sigma"].item() < _SMALL_SIGMA:
            model.change_implementation(
                non_centered=sample["sigma"].item() < _SMALL_SIGMA,
                extra_stability=needs_stability,
            )
            log.info(
                "Model implementation temporarily changed.",
                extra_stability=needs_stability,
                non_centered=sample["sigma"].item() < _SMALL_SIGMA,
            )
            posterior_sample = _sample_posterior(
                log, sample_index, sample, cache_directory, model
            )
            model.change_implementation(
                non_centered=False,
                extra_stability=False,
            )
        else:
            posterior_sample = _sample_posterior(
                log, sample_index, sample, cache_directory, model
            )

        log.info("Thinning posterior.")
        posterior_sample.thinned_sample = model.thin_posterior(
            posterior_sample.posterior
        )
        log.info("Sampling from posterior predictive distribution.")
        posterior_sample.posterior_predictive = model.sample_posterior_predictive(
            posterior_sample.thinned_sample
        )

        if (
            sample_index in [0, 1]
            or posterior_sample.thinned_sample["draw"].size < _SBC_SAMPLES / _SBC_CHAINS
        ):
            log.info(
                "Visualizing chain.",
                chain_length=posterior_sample.thinned_sample["draw"].size,
                goal_chain_length=_SBC_SAMPLES / _SBC_CHAINS,
            )
            _visualize_sample(
                model,
                figure_directory,
                sample_index,
                prior,
                sample,
                posterior_sample,
            )

        sample_sizes = summary(posterior_sample.posterior)["ess_bulk"]
        for parameter in parameters:
            if parameter in sample_sizes:
                summaries[parameter].sel(summary="sample size").values[sample_index] = (
                    sample_sizes["o"]
                )
            else:
                summaries[parameter].sel(summary="sample size").values[sample_index] = (
                    sample_sizes[
                        sample_sizes.index.str.startswith(parameter)
                    ].to_numpy()
                )

        _summarize_posterior(summaries, posterior_sample.posterior, sample_index)
        _summarize_posterior_predictive(
            predictive_summaries,
            posterior_sample.posterior_predictive,
            prior[posterior_sample.posterior_predictive.keys()].isel(draw=sample_index),
            sample_index,
        )

        log.info(
            "Posterior inference performed on data sampled from prior distribution.",
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
    log: Any,
    sample_index: int,
    sample: Dataset,
    cache_directory: Path,
    model: ThrowTimeModel,
) -> InferenceData:
    log.info("Sampling from posterior distribution.")

    match model.model_type:
        case ModelType.GAMMA:
            posterior_file = cache_directory / f"posterior_{sample_index}.nc"
        case ModelType.NAIVE:
            posterior_file = cache_directory / f"naive_posterior_{sample_index}.nc"
        case ModelType.INVGAMMA:
            posterior_file = cache_directory / f"inv_posterior_{sample_index}.nc"
        case ModelType.NAIVEINVGAMMA:
            posterior_file = cache_directory / f"naive_inv_posterior_{sample_index}.nc"

    if posterior_file.exists():
        posterior_sample = open_dataset(posterior_file)
        if len(posterior_sample.keys()) > 0:
            posterior_sample = InferenceData(posterior=posterior_sample)
            log2 = log.bind(cache_format=Dataset)
        else:
            posterior_sample = InferenceData.from_netcdf(posterior_file)
            log2 = log.bind(cache_format=InferenceData)
        log2.debug("Posterior samples loaded.", cache_file=posterior_file)
    else:
        model.change_observations(y=sample["y"].values)
        posterior_sample = model.sample(
            sample_count=_SBC_SAMPLES,
            chain_count=_SBC_CHAINS,
            parallel_count=_SBC_CHAINS,
            thin=False,
        )
        posterior_sample.to_netcdf(posterior_file)
        log.debug("Posterior samples cached.", cache_file=posterior_file)

    return posterior_sample


def _visualize_sample(  # noqa: PLR0913
    model: ThrowTimeModel,
    figure_directory: Path,
    sample_index: int,
    prior: Dataset,
    sample: Dataset,
    posterior_sample: InferenceData,
) -> None:
    match model.model_type:
        case ModelType.GAMMA:
            sample_directory = figure_directory / str(sample_index)
        case ModelType.NAIVE:
            sample_directory = figure_directory / f"naive_{sample_index}"
        case ModelType.INVGAMMA:
            sample_directory = figure_directory / f"inv_{sample_index}"
        case ModelType.NAIVEINVGAMMA:
            sample_directory = figure_directory / f"naive_inv_{sample_index}"

    data = model.dataset
    for var in posterior_sample.posterior_predictive:
        data[var] = prior[var].isel(draw=sample_index, chain=0)

    posterior_distribution_plots(
        posterior_sample.posterior,
        sample_directory / "parameters",
        prior_samples=prior,
        true_values=sample,
    )
    predictive_distributions(
        posterior_sample.posterior_predictive,
        sample_directory / "predictions",
        true_values=data,
    )
    chain_plots(
        posterior_sample.posterior,
        sample_directory / "chains",
        sample_stats=posterior_sample.get("sample_stats", None),
    )


def _summarize_posterior(
    summaries: Dataset,
    sample: Dataset,
    sample_index: int,
) -> None:
    for parameter in summaries:
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
    for parameter in summaries:
        parameter_sample = sample[parameter].values.flatten()
        prior_parameter_sample = prior_sample[parameter].values.flatten()
        summaries[parameter].sel(summary="KS distance")[sample_index] = kstest(
            parameter_sample, prior_parameter_sample
        ).statistic
