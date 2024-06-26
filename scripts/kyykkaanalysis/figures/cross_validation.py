"""
Visualizations for cross-validation.

This module provides visualizations for cross-validation and model comparisons based on
cross-validation.
"""

from pathlib import Path

import numpy as np
import structlog
from arviz import ELPDData
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from xarray import Dataset

from .utils import (
    FONT_SIZE,
    PLOT_COLORS,
    precalculated_histogram,
    uniform_variation,
)

_LOG = structlog.get_logger(__name__)


def cross_validation_plots(
    data: Dataset, loo_result: ELPDData, figure_directory: Path
) -> None:
    """
    Plot information about cross-validation.

    This function plots estimated leave-one-out log-likelihoods and estimated
    Pareto k_hat (the shape parameter of generalized Pareto distribution) values of
    importance sampling weights. It also plots the ecdf of the estimated likelihoods.

    Parameters
    ----------
    data : xarray.Dataset
        Observed data
    loo_result : arviz.ELPDData
        Results of cross-validation
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    log = _LOG.bind(figure_directory=figure_directory)
    log.info("Creating cross-validation figures.")

    figure_directory.mkdir(parents=True, exist_ok=True)

    _k_hat(data, loo_result, figure_directory)
    _log_likelihoods(data, loo_result, figure_directory)
    _log_likelihood_percentiles(loo_result, figure_directory)

    log.info("Cross-validation figures created.")


def _k_hat(data: Dataset, loo_results: ELPDData, figure_directory: Path) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["y"].values[data["is_first"]],
            y=loo_results.pareto_k.values[data["is_first"]],
            customdata=np.hstack(
                [
                    data["throws"].values.reshape(-1, 1),
                    data["player"].values.reshape(-1, 1),
                ]
            ),
            name="1. heitto",
            mode="markers",
            hovertemplate="Heittoaika: %{x} s<br>Pareto k: %{y:.2f}"
            "<br>Heiton indeksi: %{customdata[0]}<br>"
            "Heittäjän indeksi: %{customdata[1]}<extra>1. heitto</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["y"].values[~data["is_first"]],
            y=loo_results.pareto_k.values[~data["is_first"]],
            customdata=np.hstack(
                [
                    data["player"].values.reshape(-1, 1),
                    data["throws"].values.reshape(-1, 1),
                ]
            ),
            name="2. heitto",
            mode="markers",
            hovertemplate="Heittoaika: %{x} s<br>Pareto k: %{y:.2f}"
            "<br>Heiton indeksi: %{customdata[0]}<br>"
            "Heittäjän indeksi: %{customdata[1]}<extra>2. heitto</extra>",
        )
    )

    reliability_threshold = min(1 - 1 / np.log10(loo_results.n_samples), 0.7)
    figure.add_trace(
        go.Scatter(
            x=[0, data["y"].values.max() + 1],
            y=[reliability_threshold, reliability_threshold],
            name="Luotettavuusraja",
            mode="lines",
            line={"color": "black", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis={
            "title": "Heittoaika [s]",
            "range": [0, data["y"].values.max() + 1],
        },
        yaxis_title=r"$\text{Pareto }\hat{k}$",
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "k_hat.html", include_mathjax="cdn")
    _LOG.debug(
        "Pareto k-hat plot created.", figure_path=figure_directory / "k_hat.html"
    )


def _log_likelihoods(
    data: Dataset, loo_results: ELPDData, figure_directory: Path
) -> None:
    customdata = np.hstack(
        [
            data["throws"].values.reshape(-1, 1),
            data["player"].values.reshape(-1, 1),
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["y"].values[data["is_first"]],
            y=loo_results.loo_i.values[data["is_first"]],
            customdata=customdata[data["is_first"], :],
            name="1. heitto",
            mode="markers",
            hovertemplate="Heittoaika: %{x} s<br>"
            "Pisteittäinen log-uskottavuus: %{y:.2f}<br>"
            "Heiton indeksi: %{customdata[0]}<br>"
            "Heittäjän indeksi: %{customdata[1]}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["y"].values[~data["is_first"]],
            y=loo_results.loo_i.values[~data["is_first"]],
            customdata=customdata[~data["is_first"], :],
            name="2. heitto",
            mode="markers",
            hovertemplate="Heittoaika: %{x} s<br>"
            "Pisteittäinen log-uskottavuus: %{y:.2f}<br>"
            "Heiton indeksi: %{customdata[0]}<br>"
            "Heittäjän indeksi: %{customdata[1]}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Heittoaika [s]",
        yaxis_title="Pisteittäinen log-uskottavuus",
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "log_likelihood.html", include_mathjax="cdn")
    _LOG.debug(
        "Log-likelihood plot created.",
        figure_path=figure_directory / "log_likelihood.html",
    )


def _log_likelihood_percentiles(loo_results: ELPDData, figure_directory: Path) -> None:
    pit = loo_results.pit.values
    bin_count = max(min(pit.size // 20, 20), 1)
    histogram_variation, cdf_variation = uniform_variation(pit.size, bin_count)

    figure = make_subplots(rows=1, cols=2)
    figure.add_trace(histogram_variation, row=1, col=1)
    figure.add_traces(
        precalculated_histogram(pit, PLOT_COLORS[0], bin_count=bin_count),
        rows=1,
        cols=1,
    )

    figure.add_trace(cdf_variation, row=1, col=2)
    pit = np.sort(pit)
    cdf_errors = np.linspace(0, 1, pit.size) - pit
    digits = max(np.ceil(-np.log10(cdf_errors.max())).astype(int) + 1, 1)
    figure.add_trace(
        go.Scatter(
            x=pit,
            y=cdf_errors,
            mode="lines",
            line_color=PLOT_COLORS[0],
            hovertemplate="Persentiili: %{x:.3f}<br>Ero teoreettiseen: "
            f"%{{y:.{digits}f}}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="Arvioitu kertymäfunktio",
        range=[0, 1],
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Arvioitu kertymäfunktio", row=1, col=2)
    figure.update_yaxes(showticklabels=False, row=1, col=1)

    figure.update_layout(
        showlegend=False,
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "pit.html", include_mathjax="cdn")
    _LOG.debug(
        "Probability integral transform plot created.",
        figure_path=figure_directory / "pit.html",
    )


def model_comparison(
    data: Dataset, loo_results: dict[str, ELPDData], figure_directory: Path
) -> None:
    """
    Plot information about model comparisons.

    This function plots correlations of estimated leave-one-out log likelihoods from
    different models both w.r.t. each other and to the data. It also plots a comparison
    of the expected log pointwise predictive densities of different models.

    Parameters
    ----------
    data : xarray.Dataset
        Observed data
    loo_results : arviz.ELPDData
        Results of cross-validation
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    log = _LOG.bind(figure_directory=figure_directory)
    log.info("Creating model comparison figures.")

    _log_likelihood_comparison(data, loo_results, figure_directory)
    _log_likelihood_difference(data, loo_results, figure_directory)
    _elpd(loo_results, figure_directory)

    log.info("Model comparison figures created.")


def _log_likelihood_comparison(
    data: Dataset, loo_results: dict[str, ELPDData], figure_directory: Path
) -> None:
    model_names = sorted(loo_results.keys())
    customdata = np.hstack(
        [
            data["y"].values.reshape(-1, 1),
            data["throws"].values.reshape(-1, 1),
            data["player"].values.reshape(-1, 1),
        ]
    )

    figure = make_subplots(rows=len(model_names) - 1, cols=len(model_names) - 1)
    for model_index, model in enumerate(model_names[:-1]):
        model_results = loo_results[model]
        for model_index2, model2 in enumerate(model_names[model_index + 1 :]):
            model2_results = loo_results[model2]
            figure.add_trace(
                go.Scatter(
                    x=model_results.loo_i.values[data["is_first"]],
                    y=model2_results.loo_i.values[data["is_first"]],
                    customdata=customdata[data["is_first"], :],
                    mode="markers",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate=f"{model} log-uskottavuus: %{{x:.2f}}<br>{model2} "
                    "log-uskottavuus: %{y:.2f}<br>Heittoaika: %{customdata[0]} s<br>"
                    "Heiton indeksi: %{customdata[1]}<br>"
                    "Heittäjän indeksi: %{customdata[2]}<extra>1. heitto</extra>",
                ),
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            figure.add_trace(
                go.Scatter(
                    x=model_results.loo_i.values[~data["is_first"]],
                    y=model2_results.loo_i.values[~data["is_first"]],
                    customdata=customdata[~data["is_first"], :],
                    mode="markers",
                    marker_color=PLOT_COLORS[1],
                    hovertemplate=f"{model} log-uskottavuus: %{{x:.2f}}<br>{model2} "
                    "log-uskottavuus: %{y:.2f}<br>Heittoaika: %{customdata[0]} s<br>"
                    "Heiton indeksi: %{customdata[1]}<br>"
                    "Heittäjän indeksi: %{customdata[2]}<extra>2. heitto</extra>",
                ),
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            likelihood_range = [
                min(
                    model_results.loo_i.values.min(), model2_results.loo_i.values.min()
                ),
                max(
                    model_results.loo_i.values.max(), model2_results.loo_i.values.max()
                ),
            ]
            likelihood_range = [
                likelihood_range[0] - 0.1 * (likelihood_range[1] - likelihood_range[0]),
                likelihood_range[1] + 0.1 * (likelihood_range[1] - likelihood_range[0]),
            ]
            figure.add_trace(
                go.Scatter(
                    x=likelihood_range,
                    y=likelihood_range,
                    mode="lines",
                    line={"color": "black", "dash": "dash"},
                    hovertemplate="Sama log-uskottavuus",
                ),
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )

            figure.update_xaxes(
                range=likelihood_range,
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            figure.update_yaxes(
                range=likelihood_range,
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            figure.update_yaxes(
                title_text=model2,
                row=model_index + model_index2 + 1,
                col=1,
            )
        figure.update_xaxes(
            title_text=model,
            row=len(model_names) - 1,
            col=model_index + 1,
        )

    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(
        figure_directory / "log_likelihood_comparison.html", include_mathjax="cdn"
    )
    _LOG.debug(
        "Log-likelihood comparison plot created.",
        figure_path=figure_directory / "log_likelihood_comparison.html",
    )


def _log_likelihood_difference(
    data: Dataset, loo_results: dict[str, ELPDData], figure_directory: Path
) -> None:
    model_names = sorted(loo_results.keys())
    customdata = np.hstack(
        [
            data["throws"].values.reshape(-1, 1),
            data["player"].values.reshape(-1, 1),
        ]
    )

    figure = make_subplots(rows=len(model_names) - 1, cols=len(model_names) - 1)
    for model_index, model in enumerate(model_names[:-1]):
        model_results = loo_results[model]
        for model_index2, model2 in enumerate(model_names[model_index + 1 :]):
            model2_results = loo_results[model2]

            differences = model2_results.loo_i.values - model_results.loo_i.values
            figure.add_trace(
                go.Scatter(
                    x=data["y"].values[data["is_first"]],
                    y=differences[data["is_first"]],
                    customdata=customdata[data["is_first"], :],
                    mode="markers",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate="Heittoaika: %{x} s<br>"
                    "Log-uskottavuuksien ero: %{y:.2f}<br>"
                    "Heiton indeksi: %{customdata[0]}<br>"
                    "Heittäjän indeksi: %{customdata[1]}"
                    f"<extra>1. heitto<br>{model2} - {model}</extra>",
                ),
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            figure.add_trace(
                go.Scatter(
                    x=data["y"].values[data["is_first"]],
                    y=differences[~data["is_first"]],
                    customdata=customdata[~data["is_first"], :],
                    mode="markers",
                    marker_color=PLOT_COLORS[1],
                    hovertemplate="Heittoaika: %{x} s<br>"
                    "Log-uskottavuuksien ero: %{y:.2f}<br>"
                    "Heiton indeksi: %{customdata[0]}<br>"
                    "Heittäjän indeksi: %{customdata[1]}"
                    f"<extra>2. heitto<br>{model2} - {model}</extra>",
                ),
                row=model_index + model_index2 + 1,
                col=model_index + 1,
            )
            figure.update_yaxes(
                title_text="Virheiden erotus",
                row=model_index + model_index2 + 1,
                col=1,
            )
        figure.update_xaxes(
            title_text="Heittoaika [s]",
            row=len(model_names) - 1,
            col=model_index + 1,
        )

    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(
        figure_directory / "log_likelihood_differences.html", include_mathjax="cdn"
    )
    _LOG.debug(
        "Log-likelihood difference plot created.",
        figure_path=figure_directory / "log_likelihood_differences.html",
    )


def _elpd(loo_results: dict[str, ELPDData], figure_directory: Path) -> None:
    model_names = sorted(loo_results.keys(), key=lambda m: -loo_results[m].elpd_loo)
    elpd = [loo_results[model].elpd_loo for model in model_names]
    se = [loo_results[model].se for model in model_names]
    figure = go.Figure(
        go.Scatter(
            x=elpd,
            y=model_names,
            error_x={"type": "data", "array": se, "visible": True},
            mode="markers",
        )
    )
    figure.update_layout(
        xaxis_title="Pisteittäisen log-ennustetiheyden odotusarvo",
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "elpd.html", include_mathjax="cdn")
    _LOG.debug(
        "Estimated log predictive density plot created.",
        figure_path=figure_directory / "elpd.html",
    )
