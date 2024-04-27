"""
Visualizations for fake data simulations.

This module provides functions for plotting the results of fake data simulation and
simulation based calibration.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from xarray import Dataset

from .utils import (
    FONT_SIZE,
    HEATMAP_COLORS,
    PLOT_COLORS,
    parameter_to_latex,
    precalculated_histogram,
    uniform_variation,
)

_ENOUGH_FOR_HEATMAP = 1000


def estimation_plots(
    posterior_summaries: Dataset,
    posterior_predictive_summaries: Dataset,
    figure_directory: Path,
) -> None:
    """
    Plot summaries of parameter recovery.

    This function plots summaries of parameter recovery from simulated data. It plots
    the distributions of normalized and unnormalized errors, the correlations between
    the normalized errors and true parameter values, distributions of the percentiles of
    true values within the posterior samples, lengths of the thinned chains, posterior
    contractions and the Kolmogorov-Smirnov distance between the data and posterior
    predictive distributions.

    Parameters
    ----------
    posterior_summaries : xarray.Dataset
        Summaries of posterior samples relative to the true values
    posterior_predictive_summaries : xarray.Dataset
        Summaries of posterior predictive samples relative to the true values
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    figure_directory.mkdir(parents=True, exist_ok=True)

    _cm_accuracy(posterior_summaries, figure_directory)
    _error_correlations(posterior_summaries, figure_directory)
    _percentiles(posterior_summaries, figure_directory)
    _sample_sizes(posterior_summaries, figure_directory)
    _contraction(posterior_summaries, figure_directory)

    _ks_distances(posterior_summaries, posterior_predictive_summaries, figure_directory)


def _cm_accuracy(posterior_summaries: Dataset, figure_directory: Path) -> None:
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
        parameter_symbol = parameter_to_latex(parameter, variable_type="error")
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
                    hovertemplate=f"Virhe: %{{x:.2f}}{theta_hover}"
                    f"<extra>{parameter}</extra>",
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
                    hovertemplate=f"Virhe: %{{x:.2f}}{theta_hover}"
                    f"<extra>{parameter}</extra>",
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
                    hovertemplate=f"Virhe: %{{x:.2f}}{extra_hover}"
                    f"<extra>{parameter}</extra>",
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
                    hovertemplate=f"Virhe: %{{x:.2f}}{extra_hover}"
                    f"<extra>{parameter}</extra>",
                ),
                row=1,
                col=3,
            )
        figure.add_traces(
            precalculated_histogram(
                errors.flatten(), PLOT_COLORS[0], bin_count=min(errors.size // 20, 100)
            ),
            rows=parameter_index + 1,
            cols=2,
        )
        figure.add_traces(
            precalculated_histogram(
                normalized_errors.flatten(),
                PLOT_COLORS[0],
                bin_count=min(errors.size // 20, 100),
            ),
            rows=parameter_index + 1,
            cols=4,
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


def _error_correlations(posterior_summaries: Dataset, figure_directory: Path) -> None:
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
        error_symbol = parameter_to_latex(parameter, variable_type="error")
        parameter_summaries = posterior_summaries[parameter]
        errors = (
            parameter_summaries.sel(summary="conditional mean").values
            - parameter_summaries.sel(summary="true value").values
        )
        errors = (
            errors / parameter_summaries.sel(summary="posterior std").values
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
    if x.size < _ENOUGH_FOR_HEATMAP:
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
            colorscale=HEATMAP_COLORS,
            showscale=False,
            hovertemplate="Todellinen arvo: %{x}<br>"
            "Virhe: %{y}<br>Näytteitä: %{z}<extra></extra>",
        )

    return trace


def _percentiles(posterior_summaries: Dataset, figure_directory: Path) -> None:
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)

    figure = make_subplots(rows=parameter_count, cols=2)

    for parameter_index, parameter in enumerate(parameters):
        parameter_symbol = parameter_to_latex(parameter, variable_type="percentile")
        parameter_percentiles = (
            posterior_summaries[parameter].sel(summary="percentile").values.flatten()
        )
        bin_count = max(min(parameter_percentiles.size // 20, 20), 1)
        histogram_variation, cdf_variation = uniform_variation(
            parameter_percentiles.size, bin_count
        )
        figure.add_trace(
            histogram_variation,
            row=parameter_index + 1,
            col=1,
        )
        figure.add_traces(
            precalculated_histogram(
                parameter_percentiles, PLOT_COLORS[0], bin_count=bin_count
            ),
            rows=parameter_index + 1,
            cols=1,
        )

        figure.add_trace(
            cdf_variation,
            row=parameter_index + 1,
            col=2,
        )
        parameter_percentiles = np.sort(parameter_percentiles)
        cdf_errors = (
            np.linspace(0, 1, parameter_percentiles.size) - parameter_percentiles
        )
        digits = max(np.ceil(-np.log10(np.abs(cdf_errors).max())).astype(int) + 1, 1)
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


def _sample_sizes(posterior_summaries: Dataset, figure_directory: Path) -> None:
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
        sample_size = (
            posterior_summaries[parameter].sel(summary="sample size").values.flatten()
        )

        for parameter_index2, parameter2 in enumerate(parameters):
            parameter_truths = (
                posterior_summaries[parameter2].sel(summary="true value").values
            ).flatten()

            if "theta" in [parameter, parameter2]:
                figure.add_trace(
                    _sample_size_plot(
                        np.repeat(
                            parameter_truths,
                            max(sample_size.size // parameter_truths.size, 1),
                            0,
                        ),
                        np.repeat(
                            sample_size,
                            max(parameter_truths.size // sample_size.size, 1),
                            0,
                        ),
                        theta_customdata,
                        f"Otoksen koko: %{{y:.2f}}{theta_hover}"
                        f"<extra>{parameter}</extra>",
                    ),
                    row=parameter_index + 1,
                    col=parameter_index2 + 1,
                )
            else:
                figure.add_trace(
                    _sample_size_plot(
                        parameter_truths,
                        sample_size,
                        customdata,
                        f"Otoksen koko: %{{y:.2f}}{extra_hover}"
                        f"<extra>{parameter}</extra>",
                    ),
                    row=parameter_index + 1,
                    col=parameter_index2 + 1,
                )
        figure.update_yaxes(
            title_text="Otoksen koko",
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
    figure.write_html(figure_directory / "sample_sizes.html", include_mathjax="cdn")


def _sample_size_plot(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    customdata: npt.NDArray[Any],
    hover: str,
) -> go.Scatter | go.Histogram2d:
    if x.size < _ENOUGH_FOR_HEATMAP:
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
            colorscale=HEATMAP_COLORS,
            showscale=False,
            hovertemplate="Todellinen arvo: %{x}<br>"
            "Otoksen koko: %{y}<br>Näytteitä: %{z}<extra></extra>",
        )

    return trace


def _contraction(posterior_summaries: Dataset, figure_directory: Path) -> None:
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)
    col_count = int(np.ceil(np.sqrt(parameter_count)))
    row_count = int(np.ceil(parameter_count / col_count))
    figure = make_subplots(
        rows=row_count,
        cols=col_count,
        subplot_titles=[parameter_to_latex(parameter) for parameter in parameters],
    )
    customdata, theta_customdata, extra_hover, theta_hover = _simulation_infos(
        posterior_summaries, parameters
    )
    for parameter_index, parameter in enumerate(parameters):
        parameter_symbol = parameter_to_latex(parameter)
        parameter_summaries = posterior_summaries[parameter]
        mean = parameter_summaries.sel(summary="conditional mean").values.flatten()
        truth = parameter_summaries.sel(summary="true value").values.flatten()
        posterior_std = parameter_summaries.sel(
            summary="posterior std"
        ).values.flatten()
        prior_std = parameter_summaries.sel(summary="prior std").values.flatten()

        z_score = (mean - truth) / posterior_std
        contraction = 1 - np.square(posterior_std) / np.square(prior_std)

        row = parameter_index // col_count + 1
        col = parameter_index % col_count + 1
        if parameter == "theta":
            figure.add_trace(
                go.Scatter(
                    x=contraction,
                    y=z_score,
                    customdata=theta_customdata,
                    name=parameter_symbol,
                    mode="markers",
                    marker={"color": PLOT_COLORS[0], "opacity": 0.7},
                    hovertemplate=(
                        f"Posteriorin supistuma: %{{x:.2f}}<br>"
                        "Normalisoitu virhe:%{y:.2f}"
                        f"{theta_hover}<extra>{parameter}</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=contraction,
                    y=z_score,
                    customdata=customdata,
                    name=parameter_symbol,
                    mode="markers",
                    marker={"color": PLOT_COLORS[0], "opacity": 0.7},
                    hovertemplate=(
                        f"Posteriorin supistuma: %{{x:.2f}}<br>"
                        "Normalisoitu virhe:%{y:.2f}"
                        f"{extra_hover}<extra>{parameter}</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            figure.update_xaxes(
                title_text="Posteriorin supistuma",
                range=[0, 1] if contraction.min() > 0 else None,
                row=row,
                col=col,
            )
            figure.update_yaxes(title_text="Normalisoitu virhe", row=row, col=col)

    figure.update_layout(
        showlegend=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "contraction.html", include_mathjax="cdn")


def _ks_distances(
    posterior_summaries: Dataset,
    posterior_predictive_summaries: Dataset,
    figure_directory: Path,
) -> None:
    observed_variables = sorted(posterior_predictive_summaries.keys())
    parameters = sorted(posterior_summaries.keys())
    parameter_count = len(parameters)
    for variable in observed_variables:
        distance = (
            posterior_predictive_summaries[variable].sel(summary="KS distance").values
        )

        figure = make_subplots(
            rows=1,
            cols=parameter_count,
        )
        customdata, theta_customdata, extra_hover, theta_hover = _simulation_infos(
            posterior_summaries, parameters
        )
        for parameter_index, parameter in enumerate(parameters):
            parameter_symbol = parameter_to_latex(parameter)
            parameter_truths = (
                posterior_summaries[parameter].sel(summary="true value").values
            ).flatten()

            if parameter == "theta":
                figure.add_trace(
                    _sample_size_plot(
                        np.repeat(
                            parameter_truths,
                            max(distance.size // parameter_truths.size, 1),
                            0,
                        ),
                        np.repeat(
                            distance,
                            max(parameter_truths.size // distance.size, 1),
                            0,
                        ),
                        theta_customdata,
                        f"KS-testin etäisyys: %{{y:.2f}}{theta_hover}"
                        f"<extra>{parameter}</extra>",
                    ),
                    row=1,
                    col=parameter_index + 1,
                )
            else:
                figure.add_trace(
                    _sample_size_plot(
                        parameter_truths,
                        distance,
                        customdata,
                        f"KS-testin etäisyys: %{{y:.2f}}{extra_hover}"
                        f"<extra>{parameter}</extra>",
                    ),
                    row=1,
                    col=parameter_index + 1,
                )
            figure.update_yaxes(title_text="KS-testin etäisyys", row=1, col=1)
            figure.update_xaxes(
                title_text=parameter_symbol, row=1, col=parameter_index + 1
            )

        figure.update_layout(
            showlegend=False,
            separators=", ",
            font={"size": FONT_SIZE, "family": "Computer modern"},
        )
        figure.write_html(figure_directory / "ks_test.html", include_mathjax="cdn")
