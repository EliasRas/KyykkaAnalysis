"""Visualizations for prior predictive checking."""

from pathlib import Path
from typing import Any

import numpy as np
import structlog
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
)

_LOG = structlog.get_logger(__name__)


def parameter_distributions(
    samples: Dataset,
    players: npt.NDArray[np.str_],
    first_throw: npt.NDArray[np.bool_],
    figure_directory: Path,
) -> None:
    """
    Plot the distributions of the sampled parameters.

    Parameters
    ----------
    samples : xarray.Dataset
        Prior samples
    players : numpy.ndarray of str
        Names of the players for each throw
    first_throw : numpy.ndarray of bool
        Whether the throw was the player's first throw in a turn
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    log = _LOG.bind(figure_directory=figure_directory)
    log.info("Creating prior predictive figures.")

    figure_directory.mkdir(parents=True, exist_ok=True)

    _sample_distributions(samples, first_throw, figure_directory)
    _theta_ranges(samples, figure_directory)
    _throw_time_ranges(samples, figure_directory)
    _player_time_ranges(samples, players, figure_directory)

    log.info("Prior predictive figures created.")


def _sample_distributions(
    samples: Dataset, first_throw: npt.NDArray[np.bool_], figure_directory: Path
) -> None:
    log = _LOG.bind()
    for parameter, parameter_samples in samples.items():
        sample_values = parameter_samples.values
        parameter_symbol = parameter_to_latex(parameter)
        if parameter == "y":
            figure = go.Figure()
            figure.add_traces(
                precalculated_histogram(
                    sample_values[:, :, first_throw].flatten(),
                    PLOT_COLORS[0],
                    name="1. heitto",
                )
            )
            figure.add_traces(
                precalculated_histogram(
                    sample_values[:, :, ~first_throw].flatten(),
                    PLOT_COLORS[1],
                    name="2. heitto",
                )
            )
        else:
            figure = go.Figure(
                precalculated_histogram(sample_values.flatten(), PLOT_COLORS[0])
            )

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
        log.debug(
            "Distribution plot created for parameter.",
            figure_path=figure_directory / f"{parameter}.html",
        )


def _theta_ranges(samples: Dataset, figure_directory: Path) -> None:
    theta_sample = samples["theta"].values
    theta_sample = theta_sample.reshape(-1, theta_sample.shape[-1])
    minimum_thetas = theta_sample.min(1)
    maximum_thetas = theta_sample.max(1)
    figure = _range_figure(minimum_thetas, maximum_thetas, "Pelaajien keskiarvojen")
    figure.write_html(
        figure_directory / "theta_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Distribution of player's throw time mean ranges created.",
        figure_path=figure_directory / "theta_range.html",
    )


def _range_figure(
    minimum_values: npt.NDArray[Any],
    maximum_values: npt.NDArray[Any],
    values_name: str,
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


def _throw_time_ranges(samples: Dataset, figure_directory: Path) -> None:
    throw_times = samples["y"].values
    throw_times = throw_times.reshape(-1, throw_times.shape[-1])
    minimum_times = throw_times.min(1)
    maximum_times = throw_times.max(1)
    figure = _range_figure(minimum_times, maximum_times, "Heittoaikojen")
    figure.write_html(
        figure_directory / "y_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Distribution of throw time range created.",
        figure_path=figure_directory / "y_range.html",
    )


def _player_time_ranges(
    samples: Dataset, players: npt.NDArray[np.str_], figure_directory: Path
) -> None:
    throw_times = samples["y"].values
    minimum_times = []
    maximum_times = []
    for player in np.unique(players):
        from_player = players == player
        player_times = throw_times[:, :, from_player].reshape(-1, from_player.sum())
        minimum_times.extend(player_times.min(1))
        maximum_times.extend(player_times.max(1))
    minimum_times = np.array(minimum_times)
    maximum_times = np.array(maximum_times)
    figure = _range_figure(minimum_times, maximum_times, "Heittoaikojen")
    figure.write_html(
        figure_directory / "player_y_range.html",
        include_mathjax="cdn",
    )

    _LOG.debug(
        "Distribution of per player throw time range created.",
        figure_path=figure_directory / "player_y_range.html",
    )
