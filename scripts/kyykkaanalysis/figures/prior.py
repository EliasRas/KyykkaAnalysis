"""Visualizations for prior predictive checking"""
from pathlib import Path

import numpy as np
from numpy import typing as npt
from xarray import Dataset
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    parameter_to_latex,
    precalculated_histogram,
    calculate_histogram,
    PLOT_COLORS,
    FONT_SIZE,
)


def parameter_distributions(
    samples: Dataset,
    player_ids: npt.NDArray[np.str_],
    first_throw: npt.NDArray[np.bool_],
    figure_directory: Path,
) -> None:
    """
    Plot the distributions of the sampled parameters

    Parameters
    ----------
    samples : xarray.Dataset
        Prior samples
    player_ids : numpy.ndarray of str
        Names of the players for each throw
    first_throw : numpy.ndarray of bool
        Whether the throw was the player's first throw in a turn
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    figure_directory.mkdir(parents=True, exist_ok=True)

    _sample_distributions(samples, first_throw, figure_directory)
    _theta_ranges(samples, figure_directory)
    _throw_time_ranges(samples, figure_directory)
    _player_time_ranges(samples, player_ids, figure_directory)


def _sample_distributions(
    samples: Dataset, first_throw: npt.NDArray[np.bool_], figure_directory: Path
) -> None:
    for parameter, parameter_samples in samples.items():
        parameter_samples = parameter_samples.values
        parameter_symbol = parameter_to_latex(parameter)
        if parameter in ["y", "y_hat"]:
            figure = go.Figure(
                [
                    precalculated_histogram(
                        parameter_samples[:, :, first_throw].flatten()
                    ),
                    precalculated_histogram(
                        parameter_samples[:, :, ~first_throw].flatten()
                    ),
                ]
            )
            figure.data[0].name = "1. heitto"
            figure.data[1].name = "2. heitto"
            figure.update_traces(marker_opacity=0.7)
        else:
            figure = go.Figure(precalculated_histogram(parameter_samples.flatten()))

        figure.update_layout(
            xaxis_title=parameter_symbol,
            yaxis_showticklabels=False,
            barmode="overlay",
            bargap=0,
            separators=", ",
            font={"size": FONT_SIZE, "family": "Computer modern"},
        )
        figure.write_html(
            figure_directory / f"{parameter}.html",
            include_mathjax="cdn",
        )


def _theta_ranges(samples: Dataset, figure_directory: Path) -> None:
    theta_sample = samples["theta"].values
    theta_sample = theta_sample.reshape(-1, theta_sample.shape[-1])
    minimum_thetas = theta_sample.min(1)
    maximum_thetas = theta_sample.max(1)
    figure = _range_figure(
        minimum_thetas, maximum_thetas, "Pelaajan keskiarvo", "Pelaajien keskiarvojen"
    )
    figure.write_html(
        figure_directory / "theta_range.html",
        include_mathjax="cdn",
    )


def _throw_time_ranges(samples: Dataset, figure_directory: Path) -> None:
    throw_times = samples["y"].values
    throw_times = throw_times.reshape(-1, throw_times.shape[-1])
    minimum_times = throw_times.min(1)
    maximum_times = throw_times.max(1)
    figure = _range_figure(minimum_times, maximum_times, "Heittoaika", "Heittoaikojen")
    figure.write_html(
        figure_directory / "y_range.html",
        include_mathjax="cdn",
    )


def _range_figure(
    minimum_values: npt.NDArray[np.int_],
    maximum_values: npt.NDArray[np.int_],
    value_name: str,
    values_name: str,
) -> go.Figure:
    bin_count = min(minimum_values.size // 200, 200)
    figure = make_subplots(rows=2, cols=2)
    min_counts, min_bins = calculate_histogram(
        minimum_values,
        min(int(minimum_values.max() - minimum_values.min() + 1), bin_count),
    )
    figure.add_trace(
        go.Bar(
            x=min_bins[:-1] + (min_bins[1] - min_bins[0]) / 2,
            y=min_counts,
            customdata=np.hstack(
                (min_bins[:-1].reshape(-1, 1), min_bins[1:].reshape(-1, 1))
            ),
            marker={"line": {"width": 0}},
            hovertemplate=f"{value_name}: %{{customdata[0]}} - %{{customdata[1]}}<br>"
            "Osuus: %{y:.1f} %<extra></extra>",
            marker_color=PLOT_COLORS[0],
        ),
        row=1,
        col=1,
    )
    max_counts, max_bins = calculate_histogram(
        maximum_values,
        min(int(maximum_values.max() - maximum_values.min() + 1), bin_count),
    )
    figure.add_trace(
        go.Bar(
            x=max_bins[:-1] + (max_bins[1] - max_bins[0]) / 2,
            y=max_counts,
            customdata=np.hstack(
                (max_bins[:-1].reshape(-1, 1), max_bins[1:].reshape(-1, 1))
            ),
            marker={"line": {"width": 0}},
            hovertemplate=f"{value_name}: %{{customdata[0]}} - %{{customdata[1]}}<br>"
            "Osuus: %{y:.1f} %<extra></extra>",
            marker_color=PLOT_COLORS[0],
        ),
        row=1,
        col=2,
    )
    ranges = maximum_values - minimum_values
    range_counts, range_bins = calculate_histogram(
        ranges, min(int(ranges.max() - ranges.min() + 1), bin_count)
    )
    figure.add_trace(
        go.Bar(
            x=range_bins[:-1] + (range_bins[1] - range_bins[0]) / 2,
            y=range_counts,
            customdata=np.hstack(
                (range_bins[:-1].reshape(-1, 1), range_bins[1:].reshape(-1, 1))
            ),
            marker={"line": {"width": 0}},
            hovertemplate=f"{values_name} vaihteluväli: "
            "%{customdata[0]} - %{customdata[1]}<br>"
            "Osuus: %{y:.1f} %<extra></extra>",
            marker_color=PLOT_COLORS[0],
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Histogram2d(
            x=minimum_values,
            y=maximum_values,
            nbinsx=min_bins.size - 1,
            nbinsy=max_bins.size - 1,
            colorscale="thermal",
            hovertemplate=f"{values_name} minimi: %{{x}} s<br>"
            f"{values_name} maksimi: %{{y}} s<br>Näytteitä: %{{z}}<extra></extra>",
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


def _player_time_ranges(
    samples: Dataset, player_ids: npt.NDArray[np.str_], figure_directory: Path
) -> None:
    throw_times = samples["y"].values
    minimum_times = []
    maximum_times = []
    for player in np.unique(player_ids):
        from_player = player_ids == player
        player_times = throw_times[:, :, from_player].reshape(-1, from_player.sum())
        minimum_times.extend(player_times.min(1))
        maximum_times.extend(player_times.max(1))
    minimum_times = np.array(minimum_times)
    maximum_times = np.array(maximum_times)
    figure = _range_figure(minimum_times, maximum_times, "Heittoaika", "Heittoaikojen")
    figure.write_html(
        figure_directory / "player_y_range.html",
        include_mathjax="cdn",
    )
