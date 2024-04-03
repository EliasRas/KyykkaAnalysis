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
    "mu": r"$\mu$",
    "sigma": r"$\sigma$",
    "o": "$o$",
    "k": "$k$",
    "theta": r"$\theta$",
    "y": "$y$",
}
ERROR_LATEX_CONVERSION = {
    "mu": r"$\mu-\text{virhe}$",
    "sigma": r"$\sigma-\text{virhe}$",
    "o": r"$o-\text{virhe}$",
    "k": r"$k-\text{virhe}$",
    "theta": r"$\theta-\text{virhe}$",
    "y": r"$y-\text{virhe}$",
}
PERCENTILE_LATEX_CONVERSION = {
    "mu": r"$\mu-\text{persentiili}$",
    "sigma": r"$\sigma-\text{persentiili}$",
    "o": r"$o-\text{persentiili}$",
    "k": r"$k-\text{persentiili}$",
    "theta": r"$\theta-\text{persentiili}$",
    "y": r"$y-\text{persentiili}$",
}


def write_pdf(figure: go.Figure, figure_path: Path) -> None:
    """
    Saving the file twice and waiting in between to fix
    https://github.com/plotly/plotly.py/issues/3469.
    Also apply pdf specific formatting

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


def parameter_to_latex(parameter: str, variable_type: str = "variable") -> str:
    """
    Convert the name of the parameter to a LaTeX symbol

    Parameters
    ----------
    parameter : str
        Name of the parameter
    variable_type : str, default "variable"
        Type of latex string to return. Possible choices:
            "variable": Symbol of the variable
            "error": Estimation error
            "percentile":

    Returns
    -------
    str
        LaTeX symbol
    """

    if variable_type == "variable":
        symbol = LATEX_CONVERSION[parameter]
    elif variable_type == "error":
        symbol = ERROR_LATEX_CONVERSION[parameter]
    elif variable_type == "percentile":
        symbol = PERCENTILE_LATEX_CONVERSION[parameter]

    return symbol


def precalculated_histogram(
    samples: npt.NDArray[Any],
    color: str,
    name: str | None = None,
    bin_count: int | None = None,
    normalization: str = "probability",
    legendgroup: str | None = None,
) -> tuple[go.Bar, go.Scatter]:
    """
    Create a histogram out of bar graph

    Parameters
    ----------
    samples : numpy.ndarray of Any
        Samples of parameter distribution
    color : str
        Color of the trace
    name : str, optional
        Name of the trace
    bin_count : int, optional
        Number of bins
    normalization : str, default "probability"
        Normalization for the bins. One of "probability", "probability density" and "count"
    legendgroup : str, optional
        Title of the legendgroup for this trace

    Returns
    -------
    go.Bar
        Precalculated histogram
    go.Scatter
        Line visualizing the mean of the sample
    """

    if bin_count is None:
        bin_count = max(min(samples.size // 200, 200), 1)
    elif bin_count < 1:
        bin_count = 1
    counts, bins = calculate_histogram(samples, bin_count, normalization=normalization)
    if normalization == "probability":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Osuus: %{y:.1f} %"
            f"<extra>Keskiarvo: {round(samples.mean(),2)}</extra>"
        )
    elif normalization == "probability density":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Suhteellinen yleisyys: %{y:.2f} %"
            f"<extra>Keskiarvo: {round(samples.mean(),2)}</extra>"
        )
    elif normalization == "count":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Näytteet: %{y} %"
            f"<extra>{round(samples.mean(),2)}</extra>"
        )
    histogram = go.Bar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,
        y=counts,
        customdata=np.hstack((bins[:-1].reshape(-1, 1), bins[1:].reshape(-1, 1))),
        name=name,
        marker={"line": {"width": 0}, "color": color, "opacity": 0.7},
        hovertemplate=hovertemplate,
        legendgroup=legendgroup,
        legendgrouptitle_text=legendgroup,
        showlegend=name is not None,
    )
    mean = _mean_line(samples, histogram, color)

    return histogram, mean


def _mean_line(samples: npt.NDArray[Any], histogram: go.Bar, color: str) -> go.Scatter:
    mean = samples.mean()
    max_height = histogram.y.max()
    mean_line = go.Scatter(
        x=[mean, mean],
        y=[0, max_height],
        mode="lines",
        line={"color": color, "dash": "dash"},
        hoverinfo="skip",
        showlegend=False,
    )

    return mean_line


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

    bins = create_bins(values, bin_count)
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


def create_bins(values: npt.NDArray[Any], bin_count: int) -> npt.NDArray[Any]:
    """
    Create evenly distributed bins for data

    Parameters
    ----------
    values : numpy.ndarray of Any
        Values for the histogram
    bin_count : int
        Number of bins

    Returns
    -------
    numpy.ndarray of Any
        Bin edges
    """
    min_value = np.floor(values.min())
    max_value = np.ceil(values.max())
    bin_size = (max_value - min_value) / bin_count
    if np.issubdtype(values.dtype, np.integer):
        bin_size = max(round(bin_size), 1)
    bins = np.arange(min_value, max_value + bin_size, bin_size, dtype=values.dtype)
    bins = np.round(bins, 5)

    return bins


def ecdf(
    parameter_samples: npt.NDArray[Any],
    name: str | None = None,
    color: str | None = None,
    legendgroup: str | None = None,
) -> go.Scatter | tuple[go.Scatter, go.Scatter]:
    """
    Create an empirical CDF plot

    Parameters
    ----------
    parameter_samples : numpy.ndarray of Any
        Samples of parameter distribution
    name : str, optional
        Name of the trace
    color : str, optional
        Color of the trace
    legendgroup : str, optional
        Title of the legendgroup for this trace

    Returns
    -------
    go.Scatter
        Empirical CDF
    go.Scatter, optional
        Variation interval for the empirical CDF. Returned if parameter_samples has
        more than 1 dimensions
    """

    conditional_mean = np.sort(parameter_samples.flatten())
    conditional_mean = conditional_mean[np.isfinite(conditional_mean)]
    if conditional_mean.size > 10000:
        conditional_mean = conditional_mean[:: conditional_mean.size // 10000]
    cdf = go.Scatter(
        x=conditional_mean,
        y=np.linspace(0, 1, conditional_mean.size),
        mode="lines",
        name=name,
        line_color=color,
        hovertemplate="Arvo: %{x:.2f}<br>Kumulatiivinen yleisyys: %{y:.2f}<extra></extra>",
        legendgroup=legendgroup,
        legendgrouptitle_text=legendgroup,
    )

    if len(parameter_samples.shape) > 1:
        parameter_samples = np.sort(
            parameter_samples.reshape(-1, parameter_samples.shape[-1]), axis=-1
        )
        lower_bound = np.nanmin(parameter_samples, 0)
        upper_bound = np.nanmax(parameter_samples, 0)
        interval = go.Scatter(
            x=np.concatenate((lower_bound, upper_bound[::-1])),
            y=np.concatenate(
                (
                    np.linspace(0, 1, lower_bound.size),
                    np.linspace(1, 0, lower_bound.size),
                )
            ),
            fill="toself",
            fillcolor=f"rgba({','.join(hex_to_rgb(color))},0.4)",
            line_color="rgba(0,0,0,0)",
            hoverinfo="skip",
            showlegend=False,
        )

        return cdf, interval

    return cdf


def hex_to_rgb(hex: str) -> tuple[str, str, str]:
    """
    Convert hex valued color to rgb representation

    Parameters
    ----------
    hex : str
        Color in hexadecimal format

    Returns
    -------
    tuple of (str, str, str)
        Red, blue and green components of the color
    """

    hex = hex.strip("#")

    return tuple(str(int(hex[i : i + 2], 16)) for i in (0, 2, 4))
