"""Visualizations for posterior distributions"""
from pathlib import Path

import numpy as np
from numpy import typing as npt
from xarray import Dataset
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    parameter_to_latex,
    precalculated_histogram,
    PLOT_COLORS,
    FONT_SIZE,
)


def parameter_distributions(
    samples: Dataset,
    figure_directory: Path,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> None:
    """
    Plot the distributions of the sampled parameters

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

    figure_directory.mkdir(parents=True, exist_ok=True)

    _sample_distributions(
        samples, figure_directory, prior_samples=prior_samples, true_values=true_values
    )
    _parameter_correlations(samples, figure_directory, true_values=true_values)


def _sample_distributions(
    samples: Dataset,
    figure_directory: Path,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> None:
    for parameter, parameter_samples in samples.items():
        if parameter in ["k_minus"]:
            continue
        parameter_samples = parameter_samples.values
        parameter_symbol = parameter_to_latex(parameter)
        if parameter == "theta":
            figure = _theta_distributions(
                parameter_samples,
                parameter_symbol,
                prior_samples,
                true_values,
            )
        else:
            figure = _single_parameter_distribution(
                parameter,
                parameter_samples,
                parameter_symbol,
                prior_samples,
                true_values,
            )

        figure.write_html(
            figure_directory / f"{parameter}.html",
            include_mathjax="cdn",
        )


def _theta_distributions(
    samples: npt.NDArray[np.float_],
    theta_symbol: str,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> go.Figure:
    if prior_samples is not None and "theta" in prior_samples:
        prior_sample = prior_samples["theta"].values
    else:
        prior_sample = None
    if true_values is not None and "theta" in true_values:
        true_value = true_values["theta"].values[0, :]
    else:
        true_value = None
    figure = go.Figure()
    for theta_index in range(samples.shape[-1]):
        parameter_samples = samples[:, :, theta_index].flatten()
        y = f"Pelaaja {theta_index+1}"
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
                f"<extra>Pelaaja {theta_index+1}</extra>",
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
                f"<extra>Pelaaja {theta_index+1}</extra>",
            )
        )
        if true_value is not None:
            figure.add_trace(
                go.Scatter(
                    x=[true_value[theta_index]],
                    y=[y],
                    mode="markers",
                    marker={"color": "black", "size": 7},
                    hovertemplate="Todellinen arvo: %{x:.2f}"
                    f"<extra>Pelaaja {theta_index+1}</extra>",
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
                f"<extra>Pelaaja {theta_index+1}</extra>",
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
    samples: npt.NDArray[np.float_],
    parameter_symbol: str,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> go.Figure:
    figure = go.Figure(
        precalculated_histogram(
            samples.flatten(),
            name="Posteriorijakauma",
            normalization="probability density",
        )
    )
    if prior_samples is not None and parameter in prior_samples:
        figure.add_trace(
            precalculated_histogram(
                prior_samples[parameter].values.flatten(),
                name="Priorijakauma",
                normalization="probability density",
            )
        )
        figure.update_traces(opacity=0.7)
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
    true_values: Dataset | None = None,
) -> None:
    samples = samples.drop_vars(["k_minus", "theta"])
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
                    colorscale="thermal",
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


def chain_plots(
    samples: Dataset,
    figure_directory: Path,
) -> None:
    """
    Plot the distributions of the sampled parameters

    Parameters
    ----------
    samples : xarray.Dataset
        Posterior samples
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    figure_directory.mkdir(parents=True, exist_ok=True)

    _traceplot(samples, figure_directory)
    # Divergence parallel coordinates


def _traceplot(samples: Dataset, figure_directory: Path) -> None:
    samples = samples.drop_vars(["k_minus", "theta"])
    parameter_count = len(samples)

    figure = make_subplots(rows=parameter_count, cols=2)
    for parameter_index, parameter in enumerate(samples):
        parameter_symbol = parameter_to_latex(parameter)
        parameter_samples = samples[parameter].values
        for chain_index, chain in enumerate(parameter_samples):
            figure.add_trace(
                precalculated_histogram(
                    chain,
                    normalization="probability density",
                    color=PLOT_COLORS[chain_index],
                ),
                row=parameter_index + 1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    y=chain,
                    mode="lines",
                    marker={"color": PLOT_COLORS[chain_index], "opacity": 0.5},
                    hovertemplate=f"Näyte: %{{x}}<br>"
                    f"{parameter}: %{{y:.2f}}<extra></extra>",
                ),
                row=parameter_index + 1,
                col=2,
            )
        figure.update_xaxes(
            title_text=parameter_symbol,
            row=parameter_index + 1,
            col=1,
        )
        figure.update_xaxes(
            title_text="Näyte",
            row=parameter_index + 1,
            col=2,
        )
        figure.update_yaxes(
            showticklabels=False,
            row=parameter_index + 1,
            col=1,
        )
        figure.update_yaxes(
            title_text=parameter_symbol,
            row=parameter_index + 1,
            col=2,
        )
    figure.update_traces(opacity=0.5)
    figure.update_layout(
        showlegend=False,
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "chains.html", include_mathjax="cdn")
