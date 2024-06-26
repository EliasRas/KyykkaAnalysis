"""
Visualizations for posterior distributions.

This module provides functions for visualizing posterior distributions of parameters
and posterior predictive distributions of observed data.
"""

from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy import typing as npt
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew
from xarray import Dataset

from .utils import (
    FONT_SIZE,
    HEATMAP_COLORS,
    PLOT_COLORS,
    create_bins,
    ecdf,
    parameter_to_latex,
    precalculated_histogram,
)

_LOG = structlog.get_logger(__name__)


def parameter_distributions(
    samples: Dataset,
    figure_directory: Path,
    *,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> None:
    """
    Plot information about the sampled parameters.

    This function plots distributions of posterior distributions of parameters, their
    ranges and correlations. It also plots the posterior contractions of parameters if
    samples from prior distribution are provided.

    Parameters
    ----------
    samples : xarray.Dataset
        Posterior samples
    figure_directory : Path
        Path to the directory in which the figures are saved
    prior_samples : xarray.Dataset, optional
        Prior samples
    true_values : Dataset, optional
        True values of the parameters
    """

    log = _LOG.bind(
        figure_directory=figure_directory,
        prior_exist=prior_samples is not None,
        truth_exists=true_values is not None,
    )
    log.info("Creating posterior distribution figures for parameters.")

    figure_directory.mkdir(parents=True, exist_ok=True)

    _sample_distributions(
        samples, figure_directory, prior_samples=prior_samples, true_values=true_values
    )
    _parameter_correlations(samples, figure_directory, true_values=true_values)
    _theta_ranges(samples, figure_directory, true_values=true_values)
    if prior_samples is not None:
        _contraction(samples, prior_samples, figure_directory)

    log.info("Posterior distribution figures for parameters created.")


def _sample_distributions(
    samples: Dataset,
    figure_directory: Path,
    *,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> None:
    log = _LOG.bind()
    for parameter, parameter_samples in samples.items():
        sample_values = parameter_samples.values
        parameter_symbol = parameter_to_latex(parameter)
        if parameter in ["theta", "eta"]:
            figure = _per_player_distributions(
                parameter,
                sample_values,
                parameter_symbol,
                prior_samples=prior_samples,
                true_values=true_values,
            )
        else:
            figure = _single_parameter_distribution(
                parameter,
                sample_values,
                parameter_symbol,
                prior_samples=prior_samples,
                true_values=true_values,
            )

        figure.write_html(
            figure_directory / f"{parameter}.html",
            include_mathjax="cdn",
        )

        log.debug(
            "Posterior plot created for parameter.",
            figure_path=figure_directory / f"{parameter}.html",
            parameter=parameter,
        )


def _per_player_distributions(
    parameter: str,
    samples: npt.NDArray[np.float64],
    theta_symbol: str,
    *,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> go.Figure:
    if prior_samples is not None and parameter in prior_samples:
        prior_sample = prior_samples[parameter].values
    else:
        prior_sample = None
    if true_values is not None and parameter in true_values:
        true_value = true_values[parameter].values[0, :]
    else:
        true_value = None
    figure = go.Figure()
    for player_index in range(samples.shape[-1]):
        parameter_samples = samples[:, :, player_index].flatten()
        y = f"Pelaaja {player_index+1}"
        figure.add_trace(
            go.Scatter(
                x=np.linspace(parameter_samples.min(), parameter_samples.max(), 1000),
                y=[y] * 1000,
                mode="lines",
                line={"color": PLOT_COLORS[0], "width": 2},
                hovertemplate=f"Vaihteluväli: {round(parameter_samples.min(),1)}"
                f" - {round(parameter_samples.max(),1)}<br>"
                f"Kvartiiliväli: {round(np.quantile(parameter_samples, 0.25),1)} "
                f"- {round(np.quantile(parameter_samples, 0.75),1)}"
                f"<extra>Pelaaja {player_index+1}</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.linspace(
                    np.quantile(parameter_samples, 0.25),
                    np.quantile(parameter_samples, 0.75),
                    1000,
                ),
                y=[y] * 1000,
                mode="lines",
                line={"color": PLOT_COLORS[0], "width": 10},
                hovertemplate=f"Vaihteluväli: {round(parameter_samples.min(),1)}"
                f" - {round(parameter_samples.max(),1)}<br>"
                f"Kvartiiliväli: {round(np.quantile(parameter_samples, 0.25),1)} "
                f"- {round(np.quantile(parameter_samples, 0.75),1)}"
                f"<extra>Pelaaja {player_index+1}</extra>",
            )
        )
        if true_value is not None:
            figure.add_trace(
                go.Scatter(
                    x=[true_value[player_index]],
                    y=[y],
                    mode="markers",
                    marker={"color": "black", "size": 7},
                    hovertemplate="Todellinen arvo: %{x:.2f}"
                    f"<extra>Pelaaja {player_index+1}</extra>",
                )
            )

    if prior_sample is not None:
        y = "Priorijakauma"
        figure.add_trace(
            go.Scatter(
                x=np.linspace(prior_sample.min(), prior_sample.max(), 1000),
                y=[y] * 1000,
                mode="lines",
                line={"color": PLOT_COLORS[1]},
                hovertemplate=f"Vaihteluväli: {round(prior_sample.min(),1)}"
                f" - {round(prior_sample.max(),1)}<br>"
                f"Kvartiiliväli: {round(np.quantile(prior_sample, 0.25),1)} "
                f"- {round(np.quantile(prior_sample, 0.75),1)}"
                f"<extra>Priorijakauma</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.linspace(
                    np.quantile(prior_sample, 0.25),
                    np.quantile(prior_sample, 0.75),
                    1000,
                ),
                y=[y] * 1000,
                mode="lines",
                line={"color": PLOT_COLORS[1], "width": 10},
                hovertemplate=f"Vaihteluväli: {round(prior_sample.min(),1)}"
                f" - {round(prior_sample.max(),1)}<br>"
                f"Kvartiiliväli: {round(np.quantile(prior_sample, 0.25),1)} "
                f"- {round(np.quantile(prior_sample, 0.75),1)}"
                f"<extra>Pelaaja {player_index+1}</extra>",
            )
        )

    figure.update_traces(opacity=0.7)
    figure.update_layout(
        xaxis_title=theta_symbol,
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )

    return figure


def _single_parameter_distribution(
    parameter: str,
    samples: npt.NDArray[np.float64],
    parameter_symbol: str,
    *,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> go.Figure:
    figure = go.Figure(
        precalculated_histogram(
            samples.flatten(),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
        )
    )
    if prior_samples is not None and parameter in prior_samples:
        figure.add_traces(
            precalculated_histogram(
                prior_samples[parameter].values.flatten(),
                PLOT_COLORS[1],
                name="Priorijakauma",
                normalization="probability density",
            )
        )
    if true_values is not None and parameter in true_values:
        true_value = true_values[parameter].values.item()
        max_height = figure.data[0].y.max()
        figure.add_trace(
            go.Scatter(
                x=true_value * np.ones(1000),
                y=np.linspace(0, max_height, 1000),
                name="Todellinen arvo",
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            )
        )

    figure.update_layout(
        xaxis_title=parameter_symbol,
        yaxis_showticklabels=False,
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )

    return figure


def _parameter_correlations(
    samples: Dataset,
    figure_directory: Path,
    *,
    true_values: Dataset | None = None,
) -> None:
    samples = samples.drop_vars(["theta", "eta"], errors="ignore")
    parameter_count = len(samples)
    parameters = sorted(samples.keys())
    bin_count = min((samples["draw"].size * samples["chain"].size) // 200, 100)

    figure = make_subplots(rows=parameter_count - 1, cols=parameter_count - 1)
    for parameter_index, parameter in enumerate(parameters[:-1]):
        parameter_symbol = parameter_to_latex(parameter)
        parameter_samples = samples[parameter].values.flatten()
        for parameter_index2, parameter2 in enumerate(
            parameters[parameter_index + 1 :]
        ):
            parameter_symbol2 = parameter_to_latex(parameter2)
            parameter2_samples = samples[parameter2].values.flatten()
            figure.add_trace(
                go.Histogram2d(
                    x=parameter_samples,
                    y=parameter2_samples,
                    nbinsx=bin_count,
                    nbinsy=bin_count,
                    colorscale=HEATMAP_COLORS,
                    showscale=False,
                    hovertemplate=f"{parameter}: %{{x}}<br>"
                    f"{parameter2}: %{{y}}<br>Näytteitä: %{{z}}<extra></extra>",
                ),
                row=parameter_index + parameter_index2 + 1,
                col=parameter_index + 1,
            )
            if true_values is not None:
                true_value = true_values[parameter].values.item()
                true_value2 = true_values[parameter2].values.item()
                if (
                    parameter_samples.min() <= true_value <= parameter_samples.max()
                    and parameter2_samples.min()
                    <= true_value2
                    <= parameter2_samples.max()
                ):
                    figure.add_trace(
                        go.Scatter(
                            x=[true_value],
                            y=[true_value2],
                            mode="markers",
                            marker={"color": "red", "size": 7},
                            showlegend=False,
                            hovertemplate=f"{parameter}: %{{x:.2f}}<br>"
                            f"{parameter2}: %{{y:.2f}}<extra>Todelliset arvot</extra>",
                        ),
                        row=parameter_index + parameter_index2 + 1,
                        col=parameter_index + 1,
                    )
            figure.update_yaxes(
                title_text=parameter_symbol2,
                row=parameter_index + parameter_index2 + 1,
                col=1,
            )
        figure.update_xaxes(
            title_text=parameter_symbol,
            row=parameter_count - 1,
            col=parameter_index + 1,
        )

    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "correlations.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior correlation figure created.",
        figure_path=figure_directory / "correlations.html",
    )


def _theta_ranges(
    samples: Dataset, figure_directory: Path, true_values: Dataset | None = None
) -> None:
    theta_sample = samples["theta"].values
    theta_sample = theta_sample.reshape(-1, theta_sample.shape[-1])
    minimum_thetas = theta_sample.min(1)
    maximum_thetas = theta_sample.max(1)
    if true_values is not None:
        true_values = true_values["theta"].values.squeeze()

    figure = _range_figure(
        minimum_thetas,
        maximum_thetas,
        "Pelaajien keskiarvojen",
        true_values=true_values,
    )
    figure.write_html(
        figure_directory / "theta_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Player mean range figure created.",
        figure_path=figure_directory / "theta_range.html",
    )


def _range_figure(
    minimum_values: npt.NDArray[Any],
    maximum_values: npt.NDArray[Any],
    values_name: str,
    true_values: npt.NDArray[Any] | None = None,
) -> go.Figure:
    figure = make_subplots(rows=2, cols=2)
    figure.add_traces(
        precalculated_histogram(minimum_values, color=PLOT_COLORS[0]),
        rows=1,
        cols=1,
    )
    figure.add_traces(
        precalculated_histogram(maximum_values, color=PLOT_COLORS[0]),
        rows=1,
        cols=2,
    )
    ranges = maximum_values - minimum_values
    figure.add_traces(
        precalculated_histogram(ranges, color=PLOT_COLORS[0]),
        rows=2,
        cols=1,
    )
    bar_traces = [trace for trace in figure.data if isinstance(trace, go.Bar)]
    min_bins = bar_traces[0].x.size - 1
    max_bins = bar_traces[1].x.size - 1
    figure.add_trace(
        go.Histogram2d(
            x=minimum_values,
            y=maximum_values,
            nbinsx=min_bins,
            nbinsy=max_bins,
            colorscale=HEATMAP_COLORS,
            hovertemplate=f"{values_name} minimi: %{{x}} s<br>"
            f"{values_name} maksimi: %{{y}} s<br>Näytteitä: %{{z}}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    if true_values is not None:
        histogram_traces = [trace for trace in figure.data if isinstance(trace, go.Bar)]
        true_min = true_values.min()
        true_max = true_values.max()
        figure.add_trace(
            go.Scatter(
                x=true_min * np.ones(1000),
                y=np.linspace(0, histogram_traces[0].y.max(), 1000),
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=true_max * np.ones(1000),
                y=np.linspace(0, histogram_traces[1].y.max(), 1000),
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=(true_max - true_min) * np.ones(1000),
                y=np.linspace(0, histogram_traces[2].y.max(), 1000),
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        if (
            minimum_values.min() <= true_min <= minimum_values.max()
            and maximum_values.min() <= true_max <= maximum_values.max()
        ):
            figure.add_trace(
                go.Scatter(
                    x=[true_min],
                    y=[true_max],
                    mode="markers",
                    marker={"color": "red", "size": 7},
                    showlegend=False,
                    hovertemplate="Minimi %{{x:.2f}}<br>"
                    "Maksimi: %{{y:.2f}}<extra>Todelliset arvot</extra>",
                ),
                row=2,
                col=2,
            )
    _range_style(figure, values_name)

    return figure


def _range_style(figure: go.Figure, values_name: str) -> None:
    figure.update_xaxes(title_text=f"{values_name} minimi [s]", row=1, col=1)
    figure.update_xaxes(title_text=f"{values_name} maksimi [s]", row=1, col=2)
    figure.update_xaxes(title_text=f"{values_name} vaihteluväli [s]", row=2, col=1)
    figure.update_xaxes(title_text=f"{values_name} minimi [s]", row=2, col=2)
    figure.update_yaxes(showticklabels=False, row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=2)
    figure.update_yaxes(showticklabels=False, row=2, col=1)
    figure.update_yaxes(title_text=f"{values_name} maksimi [s]", row=2, col=2)
    figure.update_layout(
        showlegend=False,
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )


def _contraction(
    samples: Dataset, prior_samples: Dataset, figure_directory: Path
) -> None:
    parameters = []
    contractions = []
    for parameter, parameter_samples in samples.items():
        if parameter not in prior_samples:
            continue

        sample_values = parameter_samples.values
        prior_sample = prior_samples[parameter].values
        if parameter in ["theta", "eta"]:
            sample_values = sample_values.reshape(-1, sample_values.shape[-1])
            prior_sample = prior_sample.reshape(-1, prior_sample.shape[-1])
            parameters.extend(
                [
                    f"{parameter}_{player_index+1}"
                    for player_index in range(sample_values.shape[-1])
                ]
            )
            contractions.extend(1 - sample_values.var(0) / prior_sample.var(0))
        else:
            parameters.append(parameter)
            contractions.append(1 - sample_values.var() / prior_sample.var())

    figure = go.Figure(
        go.Scatter(
            x=contractions,
            y=np.random.default_rng().random(len(contractions)),
            customdata=parameters,
            mode="markers",
            hovertemplate="Posteriorin supistuma: %{x:.2f}"
            "<extra>%{customdata}:</extra>",
        )
    )
    figure.update_layout(
        xaxis={
            "title": "Posteriorin supistuma",
            "range": [0, 1] if min(contractions) > 0 else None,
        },
        yaxis_showticklabels=False,
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "contraction.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior contraction figure created.",
        figure_path=figure_directory / "contraction.html",
    )


def predictive_distributions(
    samples: Dataset,
    figure_directory: Path,
    *,
    true_values: Dataset | None = None,
) -> None:
    """
    Plot the information about the posterior predictive distribution of observed data.

    This function plots visualizations of the posterior predictive distributions of
    observed data and distributions of their moments and ranges.

    Parameters
    ----------
    samples : xarray.Dataset
        Posterior predictive samples
    figure_directory : Path
        Path to the directory in which the figures are saved
    true_values : xarray.Dataset, optional
        Observed data
    """

    log = _LOG.bind(
        figure_directory=figure_directory,
        truth_exists=true_values is not None,
    )
    log.info("Creating posterior predictive distribution figures.")

    figure_directory.mkdir(parents=True, exist_ok=True)

    _data_distribution(samples, figure_directory, true_values=true_values)
    _data_moments(samples, figure_directory, true_values=true_values)
    _throw_time_ranges(samples, figure_directory, true_values=true_values)
    if true_values is not None:
        _throw_data_distribution(samples, figure_directory, true_values=true_values)
        _player_data_moments(samples, figure_directory, true_values=true_values)
        _player_time_ranges(samples, figure_directory, true_values=true_values)

    log.info("Posterior predictive distribution figures created.")


def _data_distribution(
    samples: Dataset, figure_directory: Path, *, true_values: Dataset | None = None
) -> None:
    figure = make_subplots(rows=1, cols=2)

    samples = samples["y"].values.squeeze()
    figure.add_traces(
        precalculated_histogram(
            samples.flatten(),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
            legendgroup="Jakauma",
        ),
        rows=1,
        cols=1,
    )
    figure.add_traces(
        ecdf(
            samples,
            name="Posteriorijakauma",
            color=PLOT_COLORS[0],
            legendgroup="Kertymäfunktio",
        ),
        rows=1,
        cols=2,
    )

    if true_values is not None:
        true_values = true_values["y"].values.squeeze()
        figure.add_traces(
            precalculated_histogram(
                true_values.flatten(),
                PLOT_COLORS[1],
                name="Data",
                normalization="probability density",
                legendgroup="Jakauma",
            ),
            rows=1,
            cols=1,
        )

        figure.add_traces(
            ecdf(
                true_values,
                name="Data",
                color=PLOT_COLORS[1],
                legendgroup="Kertymäfunktio",
            ),
            rows=1,
            cols=2,
        )

    figure.update_xaxes(title_text=parameter_to_latex("y"), row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text=parameter_to_latex("y"), row=1, col=2)
    figure.update_yaxes(title_text="Kumulatiivinen yleisyys", row=1, col=2)
    figure.update_layout(
        legend={"groupclick": "toggleitem"},
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "y.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior predictive throw time distribution figure created.",
        figure_path=figure_directory / "y.html",
    )


def _data_moments(
    samples: Dataset, figure_directory: Path, *, true_values: Dataset | None = None
) -> None:
    figure = make_subplots(rows=2, cols=2)

    samples = samples["y"].values.reshape(-1, samples["y"].shape[-1])
    figure.add_traces(
        precalculated_histogram(
            samples.mean(0),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
        ),
        rows=1,
        cols=1,
    )
    figure.add_traces(
        precalculated_histogram(
            samples.std(0),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
        ),
        rows=1,
        cols=2,
    )
    figure.add_traces(
        precalculated_histogram(
            skew(samples, 0),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
        ),
        rows=2,
        cols=1,
    )
    figure.add_traces(
        precalculated_histogram(
            kurtosis(samples, 0),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
        ),
        rows=2,
        cols=2,
    )

    if true_values is not None:
        histogram_traces = [trace for trace in figure.data if isinstance(trace, go.Bar)]
        true_values = true_values["y"].values.flatten()

        figure.add_trace(
            go.Scatter(
                x=true_values.mean() * np.ones(1000),
                y=np.linspace(0, histogram_traces[0].y.max(), 1000),
                name="Todellinen arvo",
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f} s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=true_values.std() * np.ones(1000),
                y=np.linspace(0, histogram_traces[1].y.max(), 1000),
                name="Todellinen arvo",
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f} s<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=skew(true_values) * np.ones(1000),
                y=np.linspace(0, histogram_traces[2].y.max(), 1000),
                name="Todellinen arvo",
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=kurtosis(true_values) * np.ones(1000),
                y=np.linspace(0, histogram_traces[3].y.max(), 1000),
                name="Todellinen arvo",
                mode="lines",
                line={"color": "black", "dash": "dash"},
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    figure.update_xaxes(title_text="Keskiarvo [s]", row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text="Keskihajonta [s]", row=1, col=2)
    figure.update_yaxes(showticklabels=False, row=1, col=2)
    figure.update_xaxes(title_text="Vinous", row=2, col=1)
    figure.update_yaxes(showticklabels=False, row=2, col=1)
    figure.update_xaxes(title_text="Kurtoosi", row=2, col=2)
    figure.update_yaxes(showticklabels=False, row=2, col=2)
    figure.update_layout(
        showlegend=False,
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "y_moments.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior predictive figure created for the moments of throw time "
        "distribution.",
        figure_path=figure_directory / "y_moments.html",
    )


def _throw_time_ranges(
    samples: Dataset, figure_directory: Path, *, true_values: Dataset | None = None
) -> None:
    throw_times = samples["y"].values
    throw_times = throw_times.reshape(-1, throw_times.shape[-1])
    minimum_times = throw_times.min(1)
    maximum_times = throw_times.max(1)
    if true_values is not None:
        true_values = true_values["y"].values

    figure = _range_figure(
        minimum_times, maximum_times, "Heittoaikojen", true_values=true_values
    )
    figure.write_html(
        figure_directory / "y_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Posterior throw time range figure created.",
        figure_path=figure_directory / "y_range.html",
    )


def _throw_data_distribution(
    samples: Dataset, figure_directory: Path, *, true_values: Dataset | None = None
) -> None:
    figure = make_subplots(rows=2, cols=2)

    samples = samples["y"].values.reshape(-1, samples["y"].shape[-1])
    is_first = true_values["is_first"].values.flatten()
    true_values = true_values["y"].values.flatten()

    figure.add_traces(
        precalculated_histogram(
            samples[:, is_first].flatten(),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
            legendgroup="Jakauma, 1.",
        ),
        rows=1,
        cols=1,
    )
    figure.add_traces(
        ecdf(
            samples[:, is_first],
            name="Posteriorijakauma",
            color=PLOT_COLORS[0],
            legendgroup="Kertymäfunktio, 1.",
        ),
        rows=1,
        cols=2,
    )
    figure.add_traces(
        precalculated_histogram(
            samples[:, ~is_first].flatten(),
            PLOT_COLORS[0],
            name="Posteriorijakauma",
            normalization="probability density",
            legendgroup="Jakauma, 2.",
        ),
        rows=2,
        cols=1,
    )
    figure.add_traces(
        ecdf(
            samples[:, ~is_first],
            name="Posteriorijakauma",
            color=PLOT_COLORS[0],
            legendgroup="Kertymäfunktio, 2.",
        ),
        rows=2,
        cols=2,
    )

    figure.add_traces(
        precalculated_histogram(
            true_values[is_first],
            PLOT_COLORS[1],
            name="Data",
            normalization="probability density",
            legendgroup="Jakauma, 1.",
        ),
        rows=1,
        cols=1,
    )
    figure.add_traces(
        ecdf(
            true_values[is_first],
            name="Data",
            color=PLOT_COLORS[1],
            legendgroup="Kertymäfunktio, 1.",
        ),
        rows=1,
        cols=2,
    )
    figure.add_traces(
        precalculated_histogram(
            true_values[~is_first],
            PLOT_COLORS[1],
            name="Data",
            normalization="probability density",
            legendgroup="Jakauma, 2.",
        ),
        rows=2,
        cols=1,
    )
    figure.add_traces(
        ecdf(
            true_values[~is_first],
            name="Data",
            color=PLOT_COLORS[1],
            legendgroup="Kertymäfunktio, 2.",
        ),
        rows=2,
        cols=2,
    )

    figure.update_xaxes(title_text=parameter_to_latex("y"), row=1, col=1)
    figure.update_yaxes(title_text="1. heitto", showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text=parameter_to_latex("y"), row=1, col=2)
    figure.update_yaxes(title_text="Kumulatiivinen yleisyys", row=1, col=2)
    figure.update_xaxes(title_text=parameter_to_latex("y"), row=2, col=1)
    figure.update_yaxes(title_text="2. heitto", showticklabels=False, row=2, col=1)
    figure.update_xaxes(title_text=parameter_to_latex("y"), row=2, col=2)
    figure.update_yaxes(title_text="Kumulatiivinen yleisyys", row=2, col=2)
    figure.update_layout(
        legend={"groupclick": "toggleitem"},
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "throw_y.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior predictive per throw throw time figure created.",
        figure_path=figure_directory / "throw_y.html",
    )


def _player_data_moments(
    samples: Dataset, figure_directory: Path, *, true_values: Dataset | None = None
) -> None:
    samples = samples["y"].values.reshape(-1, samples["y"].shape[-1])
    player_ids = true_values["player"].values
    true_values = true_values["y"].values

    figure = make_subplots(rows=1, cols=2)
    for player_id in np.unique(player_ids):
        from_player = player_ids == player_id
        player_samples = samples[:, from_player]
        y = f"Pelaaja {player_id}"

        figure.add_traces(
            _player_range(player_samples.mean(1), y, true_values[from_player].mean()),
            rows=1,
            cols=1,
        )
        figure.add_traces(
            _player_range(player_samples.std(1), y, true_values[from_player].std()),
            rows=1,
            cols=2,
        )

    figure.update_traces(opacity=0.7)
    figure.update_xaxes(title_text="Keskiarvo [s]", row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text="Keskihajonta [s]", row=1, col=2)
    figure.update_yaxes(showticklabels=False, row=1, col=2)
    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "player_y_moments.html", include_mathjax="cdn")

    _LOG.debug(
        "Posterior predictive figure created for the moments of throw time "
        "distribution of players.",
        figure_path=figure_directory / "player_y_moments.html",
    )


def _player_time_ranges(
    samples: Dataset, figure_directory: Path, true_values: Dataset
) -> None:
    throw_times = samples["y"].values
    player_ids = true_values["player"].values
    true_values = true_values["y"].values

    minimum_times = []
    maximum_times = []
    true_mins = []
    true_maxes = []
    figure = make_subplots(rows=2, cols=2)
    for player_id in np.unique(player_ids):
        from_player = player_ids == player_id
        player_samples = throw_times[:, :, from_player].reshape(-1, from_player.sum())
        player_mins = player_samples.min(1)
        player_maxes = player_samples.max(1)
        minimum_times.extend(player_mins)
        maximum_times.extend(player_maxes)
        true_mins.append(true_values[from_player].min())
        true_maxes.append(true_values[from_player].max())
        y = f"Pelaaja {player_id}"

        figure.add_traces(
            _player_range(player_mins, y, true_values[from_player].min()),
            rows=1,
            cols=1,
        )
        figure.add_traces(
            _player_range(player_maxes, y, true_values[from_player].max()),
            rows=1,
            cols=2,
        )
        figure.add_traces(
            _player_range(
                player_maxes - player_mins,
                y,
                true_values[from_player].max() - true_values[from_player].min(),
            ),
            rows=2,
            cols=1,
        )

    figure.update_traces(opacity=0.7)
    figure.add_traces(
        _range_heatmap(
            np.array(minimum_times), np.array(maximum_times), true_mins, true_maxes
        ),
        rows=2,
        cols=2,
    )
    _range_style(figure, "Heittoaikojen")
    figure.write_html(
        figure_directory / "player_y_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Posterior per player throw time range figure created.",
        figure_path=figure_directory / "player_y_range.html",
    )


def _player_range(
    samples: npt.NDArray[np.int_], y: str, true_value: float
) -> tuple[go.Scatter, go.Scatter, go.Scatter]:
    total_range = go.Scatter(
        x=np.linspace(samples.min(), samples.max(), 1000),
        y=[y] * 1000,
        mode="lines",
        line={"color": PLOT_COLORS[0], "width": 2},
        hovertemplate=f"Vaihteluväli: {round(samples.min(),1)}"
        f" - {round(samples.max(),1)} s<br>"
        f"Kvartiiliväli: {round(np.quantile(samples, 0.25),1)} "
        f"- {round(np.quantile(samples, 0.75),1)} s<extra>{y}</extra>",
    )
    hdi = go.Scatter(
        x=np.linspace(
            np.quantile(samples, 0.25),
            np.quantile(samples, 0.75),
            1000,
        ),
        y=[y] * 1000,
        mode="lines",
        line={"color": PLOT_COLORS[0], "width": 10},
        hovertemplate=f"Vaihteluväli: {round(samples.min(),1)}"
        f" - {round(samples.max(),1)} s<br>"
        f"Kvartiiliväli: {round(np.quantile(samples, 0.25),1)} "
        f"- {round(np.quantile(samples, 0.75),1)} s<extra>{y}</extra>",
    )

    true_value = go.Scatter(
        x=[true_value],
        y=[y],
        mode="markers",
        marker={"color": "black", "size": 7},
        hovertemplate=f"Todellinen arvo: %{{x:.2f}} s<extra>{y}</extra>",
    )

    return total_range, hdi, true_value


def _range_heatmap(
    minimum_times: npt.NDArray[np.int_],
    maximum_times: npt.NDArray[np.int_],
    true_mins: list[int],
    true_maxes: list[int],
) -> tuple[go.Histogram2d, go.Scatter]:
    min_bins = create_bins(minimum_times, max(min(minimum_times.size // 200, 200), 1))
    max_bins = create_bins(maximum_times, max(min(maximum_times.size // 200, 200), 1))

    heatmap = go.Histogram2d(
        x=minimum_times,
        y=maximum_times,
        xbins={
            "start": min_bins[0],
            "end": min_bins[-1],
            "size": min_bins[1] - min_bins[0],
        },
        ybins={
            "start": max_bins[0],
            "end": max_bins[-1],
            "size": max_bins[1] - max_bins[0],
        },
        colorscale=HEATMAP_COLORS,
        hovertemplate="Heittoaikojen minimi: %{x} s<br>"
        "Heittoaikojen maksimi: %{y} s<br>Näytteitä: %{z}<extra></extra>",
    )
    true_values = go.Scatter(
        x=true_mins,
        y=true_maxes,
        mode="markers",
        marker={"color": "red", "size": 7},
        showlegend=False,
        hovertemplate="Heittoaikojen minimi: %{x:.2f}<br>"
        "Heittoaikojen maksimi: %{y:.2f}<extra>Todelliset arvot</extra>",
    )

    return heatmap, true_values
