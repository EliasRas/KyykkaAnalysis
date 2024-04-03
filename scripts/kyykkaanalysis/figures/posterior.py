"""Visualizations for posterior distributions"""

from typing import Any
from pathlib import Path

import numpy as np
from numpy import typing as npt
from scipy.stats import skew, kurtosis
from xarray import Dataset
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    parameter_to_latex,
    precalculated_histogram,
    create_bins,
    ecdf,
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
    _theta_ranges(samples, figure_directory, true_values=true_values)


def _sample_distributions(
    samples: Dataset,
    figure_directory: Path,
    prior_samples: Dataset | None = None,
    true_values: Dataset | None = None,
) -> None:
    for parameter, parameter_samples in samples.items():
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
    true_values: Dataset | None = None,
) -> None:
    samples = samples.drop_vars(["theta"])
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
            colorscale="thermal",
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


def predictive_distributions(
    samples: Dataset,
    figure_directory: Path,
    true_values: Dataset | None = None,
) -> None:
    """
    Plot the distributions of the sampled parameters

    Parameters
    ----------
    sample : xarray.Dataset
        Posterior predictive samples
    figure_directory : Path
        Path to the directory in which the figures are saved
    true_values : xarray.Dataset, optional
        Observed data
    """

    _data_distribution(samples, figure_directory, true_values=true_values)
    _data_moments(samples, figure_directory, true_values=true_values)
    _throw_time_ranges(samples, figure_directory, true_values=true_values)
    if true_values is not None:
        _player_time_ranges(samples, figure_directory, true_values)


def _data_distribution(
    samples: Dataset, figure_directory: Path, true_values: Dataset | None = None
):
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
                "Data",
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


def _data_moments(
    samples: Dataset, figure_directory: Path, true_values: Dataset | None = None
):
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
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
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
                hovertemplate="Todellinen arvo: %{x:.2f}<extra></extra>",
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

    figure.update_xaxes(title_text="Keskiarvo", row=1, col=1)
    figure.update_yaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(title_text="Keskihajonta", row=1, col=2)
    figure.update_yaxes(showticklabels=False, row=1, col=2)
    figure.update_xaxes(title_text="Vinous", row=2, col=1)
    figure.update_yaxes(showticklabels=False, row=2, col=1)
    figure.update_xaxes(title_text="Huipukkuus", row=2, col=2)
    figure.update_yaxes(showticklabels=False, row=2, col=2)
    figure.update_layout(
        showlegend=False,
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "y_moments.html", include_mathjax="cdn")


def _throw_time_ranges(
    samples: Dataset, figure_directory: Path, true_values: Dataset | None = None
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


def _player_time_ranges(
    samples: Dataset,
    figure_directory: Path,
    true_values: Dataset,
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


def _player_range(
    samples: npt.NDArray[np.int_], y: str, true_value: float
) -> tuple[go.Scatter, go.Scatter, go.Scatter]:
    total_range = go.Scatter(
        x=np.linspace(samples.min(), samples.max(), 1000),
        y=[y] * 1000,
        mode="lines",
        line={"color": PLOT_COLORS[0], "width": 2},
        hovertemplate=f"Vaihteluväli: {round(samples.min(),1)}"
        f" - {round(samples.max(),1)}<br>"
        f"Kvartiiliväli: {round(np.quantile(samples, 0.25),1)} "
        f"- {round(np.quantile(samples, 0.75),1)}<extra>{y}</extra>",
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
        f" - {round(samples.max(),1)}<br>"
        f"Kvartiiliväli: {round(np.quantile(samples, 0.25),1)} "
        f"- {round(np.quantile(samples, 0.75),1)}<extra>{y}</extra>",
    )

    true_value = go.Scatter(
        x=[true_value],
        y=[y],
        mode="markers",
        marker={"color": "black", "size": 7},
        hovertemplate="Todellinen arvo: %{x:.2f}" f"<extra>{y}</extra>",
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
        colorscale="thermal",
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
    samples = samples.drop_vars(["theta"])
    parameter_count = len(samples)

    figure = make_subplots(rows=parameter_count, cols=2)
    for parameter_index, parameter in enumerate(samples):
        parameter_symbol = parameter_to_latex(parameter)
        parameter_samples = samples[parameter].values
        for chain_index, chain in enumerate(parameter_samples):
            figure.add_traces(
                precalculated_histogram(
                    chain,
                    PLOT_COLORS[chain_index],
                    normalization="probability density",
                    bin_count=min(parameter_samples.size // 100, 200),
                ),
                rows=parameter_index + 1,
                cols=1,
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
