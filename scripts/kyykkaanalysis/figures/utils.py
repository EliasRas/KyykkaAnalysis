"""Utility functions for plotly figures"""
from typing import Any
from pathlib import Path
from time import sleep

import numpy as np
from numpy import typing as npt
from plotly import graph_objects as go, colors

PLOT_COLORS = colors.qualitative.Plotly
FONT_SIZE_2X2 = 28
FONT_SIZE_BOXPLOT = 10
FONT_SIZE = 15
LATEX_CONVERSION = {
    "mu": "$\mu$",
    "sigma": "$\sigma$",
    "o": "$o$",
    "k": "$k$",
    "theta": r"$\theta$",
    "y_hat": "$\hat{y}$",
    "y": "$y$",
}


def write_pdf(figure: go.Figure, figure_path: Path) -> None:
    """
    Saving the file twice and waiting in between to fix
    https://github.com/plotly/plotly.py/issues/3469.
    In addition apply pdf specific formatting

    Parameters
    ----------
    figure : go.Figure
        Figure
    figure_path : Path
        Path to the file in which the figure is saved.
    """

    try:
        (
            rows,
            cols,
        ) = figure._get_subplot_rows_columns()  # pylint: disable=protected-access
        rows = list(rows)
        cols = list(cols)
        for row in rows:
            for col in cols:
                figure.update_xaxes(showline=True, showgrid=False, row=row, col=col)
                figure.update_yaxes(showline=True, showgrid=False, row=row, col=col)
    except Exception:  # pylint: disable=broad-exception-caught
        figure.update_layout(
            xaxis_showline=True,
            xaxis_showgrid=False,
            yaxis_showline=True,
            yaxis_showgrid=False,
        )
        rows = []
        cols = []
    if any(isinstance(data, go.Histogram) for data in figure.data):
        figure.update_traces(marker={"line": {"width": 1}})
    figure.update_layout(plot_bgcolor="white", bargap=0)

    figure.write_image(figure_path)
    sleep(1)
    figure.write_image(figure_path)

    for row in rows:
        for col in cols:
            figure.update_xaxes(showline=None, showgrid=None, row=row, col=col)
            figure.update_yaxes(showline=None, showgrid=None, row=row, col=col)
    figure.update_layout(
        plot_bgcolor=None,
        bargap=None,
        xaxis_showline=None,
        xaxis_showgrid=None,
        yaxis_showline=None,
        yaxis_showgrid=None,
    )


def parameter_to_latex(parameter: str) -> str:
    """
    Convert the name of the parameter to a LaTeX symbol

    Parameters
    ----------
    parameter : str
        Name of the parameter

    Returns
    -------
    str
        LaTeX symbol
    """

    return LATEX_CONVERSION[parameter]


def precalculated_histogram(
    parameter_samples: npt.NDArray[Any],
    name: str | None = None,
    color: str | None = None,
    normalization: str = "probability",
) -> go.Bar:
    """
    Create a histogram out of bar graph

    Parameters
    ----------
    parameter_samples : numpy.ndarray of Any
        Samples of parameter distribution
    name : str, optional
        Name of the trace
    color : str, optional
        Color of the trace
    normalization : str, default "probability"
        Normalization for the bins. One of "probability", "probability density" and "count"

    Returns
    -------
    go.Bar
        Precalculated histogram
    """
    bin_count = min(parameter_samples.size // 200, 200)
    counts, bins = calculate_histogram(
        parameter_samples, bin_count, normalization=normalization
    )
    if normalization == "probability":
        hovertemplate = (
            "Arvo: %{customdata[0]:.1f} - %{customdata[1]:.1f}<br>"
            "Osuus: %{y:.1f} %<extra></extra>"
        )
    elif normalization == "probability density":
        hovertemplate = (
            "Arvo: %{customdata[0]:.1f} - %{customdata[1]:.1f}<br>"
            "Suhteellinen yleisyys: %{y:.1f} %<extra></extra>"
        )
    elif normalization == "count":
        hovertemplate = (
            "Arvo: %{customdata[0]:.1f} - %{customdata[1]:.1f}<br>"
            "Näytteet: %{y:.1f} %<extra></extra>"
        )
    histogram = go.Bar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,
        y=counts,
        customdata=np.hstack((bins[:-1].reshape(-1, 1), bins[1:].reshape(-1, 1))),
        name=name,
        marker={"line": {"width": 0}, "color": color},
        hovertemplate=hovertemplate,
    )

    return histogram


def calculate_histogram(
    values: npt.NDArray[Any], bin_count: int, normalization: str = "probability"
) -> tuple[npt.NDArray[np.float_], npt.NDArray[Any]]:
    """
    Calculate bins and bin counts of a histogram

    Parameters
    ----------
    values : numpy.ndarray of Any
        Values for the histogram
    bin_count : int
        Number of bins
    normalization : str, default "probability"
        Normalization for the bins. One of "probability", "probability density" and "count"

    Returns
    -------
    numpy.ndarray of float
        Possibly normalized bin heights
    numpy.ndarray of Any
        Bin edges
    """
    min_value = np.floor(values.min())
    max_value = np.ceil(values.max())
    bin_size = (max_value - min_value) / bin_count
    if np.issubdtype(values.dtype, np.integer):
        bin_size = round(bin_size)
    bins = np.arange(min_value, max_value + bin_size, bin_size, dtype=values.dtype)
    bins = np.round(bins, 5)

    counts, _ = np.histogram(values, bins)
    if normalization == "probability":
        counts = counts / counts.sum() * 100
    elif normalization == "probability density":
        area = (bins[1] - bins[0]) * counts.sum()
        counts = counts / area
    elif normalization == "count":
        pass
    else:
        raise ValueError(f"Invalid normalization {normalization}.")

    return counts, bins
