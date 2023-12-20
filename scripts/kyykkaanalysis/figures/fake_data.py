"""Visualizations for fake data simulations"""
from typing import Any
from pathlib import Path

import numpy as np
from numpy import typing as npt
from xarray import Dataset
from scipy.stats import binom, beta
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    parameter_to_latex,
    precalculated_histogram,
    PLOT_COLORS,
    FONT_SIZE,
)


def estimation_plots(
    posterior_summaries: Dataset,
    figure_directory: Path,
) -> None:
    """
    Plot the distributions of the sampled parameters

    Parameters
    ----------
    posterior_summaries : xarray.Dataset
        Summaries of posterior samples relative to the true values
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    figure_directory.mkdir(parents=True, exist_ok=True)

    _cm_accuracy(posterior_summaries, figure_directory)
    _percentiles(posterior_summaries, figure_directory)
    _error_correlations(posterior_summaries, figure_directory)


def _cm_accuracy(
    posterior_summaries: Dataset,
    figure_directory: Path,
) -> None:
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)

    figure = make_subplots(
        rows=parameter_count,
        cols=4,
        subplot_titles=["Virhe", "Virhe", "Normalisoitu virhe", "Normalisoitu virhe"],
        specs=[[{"rowspan": parameter_count}, {}, {"rowspan": parameter_count}, {}]]
        + [[None, {}, None, {}] for _ in range(1, parameter_count)],
    )
    customdata, theta_customdata, extra_hover, theta_hover = _simulation_infos(
        posterior_summaries, parameters
    )

    for parameter_index, parameter in enumerate(parameters):
        parameter_symbol = parameter_to_latex(parameter, type="error")
        parameter_summaries = posterior_summaries[parameter]

        errors = (
            parameter_summaries.sel(summary="conditional mean").values
            - parameter_summaries.sel(summary="true value").values
        )
        normalized_errors = (
            errors / parameter_summaries.sel(summary="posterior std").values
        )
        if parameter == "theta":
            figure.add_trace(
                go.Box(
                    x=errors.flatten(),
                    customdata=theta_customdata,
                    name=parameter_symbol,
                    boxpoints="all",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate=f"Virhe: %{{x:.2f}}{theta_hover}<extra>{parameter}</extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Box(
                    x=normalized_errors.flatten(),
                    name=parameter_symbol,
                    customdata=theta_customdata,
                    boxpoints="all",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate=f"Virhe: %{{x:.2f}}{theta_hover}<extra>{parameter}</extra>",
                ),
                row=1,
                col=3,
            )
        else:
            figure.add_trace(
                go.Box(
                    x=errors.flatten(),
                    customdata=customdata,
                    name=parameter_symbol,
                    boxpoints="all",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate=f"Virhe: %{{x:.2f}}{extra_hover}<extra>{parameter}</extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Box(
                    x=normalized_errors.flatten(),
                    name=parameter_symbol,
                    customdata=customdata,
                    boxpoints="all",
                    marker_color=PLOT_COLORS[0],
                    hovertemplate=f"Virhe: %{{x:.2f}}{extra_hover}<extra>{parameter}</extra>",
                ),
                row=1,
                col=3,
            )
        figure.add_trace(
            precalculated_histogram(
                errors.flatten(),
                color=PLOT_COLORS[0],
                bin_count=min(errors.size // 20, 100),
            ),
            row=parameter_index + 1,
            col=2,
        )
        figure.add_trace(
            precalculated_histogram(
                normalized_errors.flatten(),
                color=PLOT_COLORS[0],
                bin_count=min(errors.size // 20, 100),
            ),
            row=parameter_index + 1,
            col=4,
        )

        figure.update_xaxes(title_text=parameter_symbol, row=parameter_index + 1, col=2)
        figure.update_xaxes(
            title_text=parameter_symbol,
            row=parameter_index + 1,
            col=4,
        )
        figure.update_yaxes(showticklabels=False, row=parameter_index + 1, col=2)
        figure.update_yaxes(showticklabels=False, row=parameter_index + 1, col=4)
    figure.update_yaxes(autorange="reversed", row=1, col=1)
    figure.update_yaxes(autorange="reversed", row=1, col=3)
    figure.update_layout(
        showlegend=False,
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "errors.html", include_mathjax="cdn")


def _simulation_infos(
    posterior_summaries: Dataset, parameters: list[str]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], str, str]:
    scalar_parameters = [parameter for parameter in parameters if parameter != "theta"]
    parameter_count = len(parameters)

    customdata = np.hstack(
        [posterior_summaries.coords["draw"].values.reshape(-1, 1)]
        + [
            posterior_summaries[parameter]
            .sel(summary="true value")
            .values.reshape(-1, 1)
            for parameter in scalar_parameters
        ]
    )
    true_thetas = (
        posterior_summaries["theta"].sel(summary="true value").values.reshape(-1, 1)
    )
    theta_customdata = np.hstack(
        (
            np.repeat(
                customdata,
                true_thetas.shape[0] // customdata.shape[0],
                0,
            ),
            true_thetas,
        )
    )
    parameter_hovers = [
        f"Todellinen {parameter}: %{{customdata[{parameter_index+1}]:.2f}}"
        for parameter_index, parameter in enumerate(scalar_parameters)
    ]
    extra_hover = (
        f"<br>Simulaatio: %{{customdata[0]}}<br>"
        f"{'<br>'.join(parameter_hovers)}<extra></extra>"
    )
    theta_hover = (
        f"<br>Simulaatio: %{{customdata[0]}}<br>{'<br>'.join(parameter_hovers)}<br>"
        f"Todellinen theta: %{{customdata[{parameter_count}]:.2f}}<extra></extra>"
    )

    return customdata, theta_customdata, extra_hover, theta_hover


def _error_correlations(
    posterior_summaries: Dataset,
    figure_directory: Path,
) -> None:
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)

    figure = make_subplots(
        rows=parameter_count,
        cols=parameter_count,
    )
    customdata, theta_customdata, extra_hover, theta_hover = _simulation_infos(
        posterior_summaries, parameters
    )
    for parameter_index, parameter in enumerate(parameters):
        parameter_symbol = parameter_to_latex(parameter)
        error_symbol = parameter_to_latex(parameter, type="error")
        parameter_summaries = posterior_summaries[parameter]
        errors = (
            parameter_summaries.sel(summary="conditional mean").values
            - parameter_summaries.sel(summary="true value").values
        ).flatten()

        for parameter_index2, parameter2 in enumerate(parameters):
            parameter_truths = (
                posterior_summaries[parameter2].sel(summary="true value").values
            ).flatten()

            if "theta" in [parameter, parameter2]:
                figure.add_trace(
                    _error_correlation_plot(
                        np.repeat(
                            parameter_truths,
                            max(errors.size // parameter_truths.size, 1),
                            0,
                        ),
                        np.repeat(
                            errors,
                            max(parameter_truths.size // errors.size, 1),
                            0,
                        ),
                        theta_customdata,
                        f"Virhe: %{{y:.2f}}{theta_hover}<extra>{parameter}</extra>",
                    ),
                    row=parameter_index + 1,
                    col=parameter_index2 + 1,
                )
            else:
                figure.add_trace(
                    _error_correlation_plot(
                        parameter_truths,
                        errors,
                        customdata,
                        f"Virhe: %{{y:.2f}}{extra_hover}<extra>{parameter}</extra>",
                    ),
                    row=parameter_index + 1,
                    col=parameter_index2 + 1,
                )
        figure.update_yaxes(
            title_text=error_symbol,
            row=parameter_index + 1,
            col=1,
        )
        figure.update_xaxes(
            title_text=parameter_symbol,
            row=parameter_count,
            col=parameter_index + 1,
        )

    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(
        figure_directory / "error_correlations.html", include_mathjax="cdn"
    )


def _error_correlation_plot(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    customdata: npt.NDArray[Any],
    hover: str,
) -> go.Scatter | go.Histogram2d:
    if x.size < 1000:
        trace = go.Scatter(
            x=x,
            y=y,
            customdata=customdata,
            mode="markers",
            marker_color=PLOT_COLORS[0],
            hovertemplate=hover,
        )
    else:
        bin_count = min(x.size // 50, 200)
        trace = go.Histogram2d(
            x=x,
            y=y,
            nbinsx=bin_count,
            nbinsy=bin_count,
            colorscale="thermal",
            showscale=False,
            hovertemplate="Todellinen arvo: %{x}<br>"
            "Virhe: %{y}<br>Näytteitä: %{z}<extra></extra>",
        )

    return trace


def _percentiles(
    posterior_summaries: Dataset,
    figure_directory: Path,
) -> None:
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)

    figure = make_subplots(rows=parameter_count, cols=2)

    for parameter_index, parameter in enumerate(parameters):
        parameter_symbol = parameter_to_latex(parameter, type="percentile")
        parameter_percentiles = (
            posterior_summaries[parameter].sel(summary="percentile").values.flatten()
        )
        bin_count = min(parameter_percentiles.size // 20, 20)
        _percentile_variation(parameter_percentiles, bin_count, figure, parameter_index)
        figure.add_trace(
            precalculated_histogram(
                parameter_percentiles, color=PLOT_COLORS[0], bin_count=bin_count
            ),
            row=parameter_index + 1,
            col=1,
        )

        parameter_percentiles = np.sort(parameter_percentiles)
        cdf_errors = (
            np.linspace(0, 1, parameter_percentiles.size) - parameter_percentiles
        )
        digits = max(np.ceil(-np.log10(cdf_errors.max())).astype(int) + 1, 1)
        figure.add_trace(
            go.Scatter(
                x=parameter_percentiles,
                y=cdf_errors,
                mode="lines",
                line_color=PLOT_COLORS[0],
                hovertemplate="Persentiili: %{x:.3f}<br>Ero teoreettiseen: "
                f"%{{y:.{digits}f}}<extra></extra>",
            ),
            row=parameter_index + 1,
            col=2,
        )

        figure.update_xaxes(
            title_text=parameter_symbol, range=[0, 1], row=parameter_index + 1, col=1
        )
        figure.update_xaxes(title_text=parameter_symbol, row=parameter_index + 1, col=2)
        figure.update_yaxes(showticklabels=False, row=parameter_index + 1, col=1)

    figure.update_layout(
        showlegend=False,
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "percentiles.html", include_mathjax="cdn")


def _percentile_variation(
    percentiles: npt.NDArray[np.float_],
    bin_count: int,
    figure: go.Figure,
    parameter_index: int,
) -> None:
    # Bin counts are binomially distributed
    bar_dist = binom(percentiles.size, 1 / bin_count)
    bar_bounds = [
        bar_dist.ppf(0.005) / percentiles.size * 100,
        bar_dist.ppf(0.995) / percentiles.size * 100,
    ]
    figure.add_trace(
        go.Scatter(
            x=[0, 1, 1, 0],
            y=[bar_bounds[0], bar_bounds[0], bar_bounds[1], bar_bounds[1]],
            fill="toself",
            fillcolor="rgba(0,0,0,0.2)",
            line_color="rgba(0,0,0,0)",
            hoverinfo="skip",
        ),
        row=parameter_index + 1,
        col=1,
    )

    # Percentiles are a uniform variable on [0,1 ], whose kth order statistic (basically
    # the inverse CDF) is beta-distributed
    lower_bound = np.linspace(0, 1, 1000)
    higher_bound = lower_bound.copy()
    left_bound = []
    right_bound = []
    for percentile_index in range(lower_bound.size):
        rank = percentile_index / lower_bound.size * percentiles.size
        cdf_dist = beta(rank, percentiles.size + 1 - rank)
        left_bound.append(cdf_dist.ppf(0.005))
        right_bound.append(cdf_dist.ppf(0.995))
        higher_bound[percentile_index] -= left_bound[-1]
        lower_bound[percentile_index] -= right_bound[-1]
    figure.add_trace(
        go.Scatter(
            x=left_bound + right_bound[::-1],
            y=np.concatenate((higher_bound, lower_bound[::-1])),
            fill="toself",
            fillcolor="rgba(0,0,0,0.2)",
            line_color="rgba(0,0,0,0)",
            hoverinfo="skip",
        ),
        row=parameter_index + 1,
        col=2,
    )
