"""
Utilities for plotly figures.

This module provides various utility functions and variables for plotly figures
including file io wrappers and generic plots
"""

from pathlib import Path
from time import sleep
from typing import Any, Literal

import numpy as np
from numpy import typing as npt
from plotly import colors
from plotly import graph_objects as go
from scipy.stats import beta, binom

FONT_SIZE = 15
"""Size of the font for figures meant to be inserted directly to report PDF"""
FONT_SIZE_2X2 = 28
"""
Size of the font for figures meant to be inserted in a 2x2 subfigure grid to report PDF
"""
FONT_SIZE_BOXPLOT = 10
"""Size of the font for boxplot figures in report PDF"""
PLOT_COLORS = colors.qualitative.Plotly
"""Default colorcycle"""

_LATEX_CONVERSION = {
    "mu": r"$\mu$",
    "sigma": r"$\sigma$",
    "o": "$o$",
    "k": "$k$",
    "a": "$a$",
    "theta": r"$\theta$",
    "eta": r"$\eta$",
    "y": "$y$",
}
_ERROR_LATEX_CONVERSION = {
    "mu": r"$\mu-\text{virhe}$",
    "sigma": r"$\sigma-\text{virhe}$",
    "o": r"$o-\text{virhe}$",
    "k": r"$k-\text{virhe}$",
    "a": r"$a-\text{virhe}$",
    "theta": r"$\theta-\text{virhe}$",
    "eta": r"$\eta-\text{virhe}$",
    "y": r"$y-\text{virhe}$",
}
_PERCENTILE_LATEX_CONVERSION = {
    "mu": r"$\mu-\text{prosenttipiste}$",
    "sigma": r"$\sigma-\text{prosenttipiste}$",
    "o": r"$o-\text{prosenttipiste}$",
    "k": r"$k-\text{prosenttipiste}$",
    "a": r"$a-\text{prosenttipiste}$",
    "theta": r"$\theta-\text{prosenttipiste}$",
    "eta": r"$\eta-\text{prosenttipiste}$",
    "y": r"$y-\text{prosenttipiste}$",
}
_TOO_MANY_SCATTER = 10000


def write_pdf(figure: go.Figure, figure_path: Path) -> None:
    """
    Write plotly figure to a PDF file.

    Saving the file twice and waiting in between to fix
    https://github.com/plotly/plotly.py/issues/3469.
    Also apply pdf specific formatting.

    Parameters
    ----------
    figure : go.Figure
        Figure
    figure_path : Path
        Path to the file in which the figure is saved.
    """

    try:
        rows, cols = figure._get_subplot_rows_columns()  # noqa: SLF001
        rows = list(rows)
        cols = list(cols)
        for row in rows:
            for col in cols:
                figure.update_xaxes(showline=True, showgrid=False, row=row, col=col)
                figure.update_yaxes(showline=True, showgrid=False, row=row, col=col)
    except Exception:  # noqa: BLE001
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


def parameter_to_latex(
    parameter: str,
    *,
    variable_type: Literal["variable", "error", "percentile"] = "variable",
) -> str:
    """
    Convert the name of a variable to a LaTeX symbol.

    Returns a LaTeX string of a variable, it's estimation error or the percentile of a
    value in sample.

    Parameters
    ----------
    parameter : str
        Name of the parameter
    variable_type : str, default "variable"
        Type of latex string to return. Possible choices:
            "variable": Symbol of the variable
            "error": Estimation error
            "percentile": Percentile of variable

    Returns
    -------
    str
        LaTeX symbol
    """

    if variable_type == "variable":
        symbol = _LATEX_CONVERSION[parameter]
    elif variable_type == "error":
        symbol = _ERROR_LATEX_CONVERSION[parameter]
    elif variable_type == "percentile":
        symbol = _PERCENTILE_LATEX_CONVERSION[parameter]

    return symbol


def precalculated_histogram(  # noqa: PLR0913
    values: npt.NDArray[Any],
    color: str,
    *,
    name: str | None = None,
    bin_count: int | None = None,
    normalization: Literal[
        "probability", "probability density", "count"
    ] = "probability",
    legendgroup: str | None = None,
) -> tuple[go.Bar, go.Scatter]:
    """
    Create a histogram out of bar graph.

    Creates a bar graph which shows the same information as a plotly histogram without
    storing the raw data with the graph. Helps improve performance and size of the
    figures.

    Parameters
    ----------
    values : numpy.ndarray of Any
        Values for the histogram
    color : str
        Color of the trace
    name : str, optional
        Name of the trace
    bin_count : int, optional
        Number of bins
    normalization : str, default "probability"
        Normalization for the bins. One of "probability", "probability density" and
        "count"
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
        bin_count = max(min(values.size // 200, 200), 1)
    elif bin_count < 1:
        bin_count = 1
    counts, bins = calculate_histogram(values, bin_count, normalization=normalization)
    if normalization == "probability":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Osuus: %{y:.1f} %"
            f"<extra>{f'{name}<br>' if name is not None else ''}"
            f"Keskiarvo: {round(values.mean(),2)}</extra>"
        )
    elif normalization == "probability density":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Suhteellinen yleisyys: %{y:.2f} %"
            f"<extra>{f'{name}<br>' if name is not None else ''}"
            f"Keskiarvo: {round(values.mean(),2)}</extra>"
        )
    elif normalization == "count":
        hovertemplate = (
            "Arvo: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>"
            "Näytteet: %{y} %"
            f"<extra>{f'{name}<br>' if name is not None else ''}"
            f"Keskiarvo: {round(values.mean(),2)}</extra>"
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
    mean = _mean_line(values, histogram, color)

    return histogram, mean


def calculate_histogram(
    values: npt.NDArray[Any],
    bin_count: int,
    *,
    normalization: Literal[
        "probability", "probability density", "count"
    ] = "probability",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
    """
    Calculate bins and bin counts of a histogram.

    Calculates histogram bins and the bin counts while accounting for data containing
    only integers or no variance.

    Parameters
    ----------
    values : numpy.ndarray of Any
        Values for the histogram
    bin_count : int
        Number of bins
    normalization : str, default "probability"
        Normalization for the bins. One of "probability", "probability density" and
        "count"

    Returns
    -------
    numpy.ndarray of float
        Possibly normalized bin heights
    numpy.ndarray of Any
        Bin edges

    Raises
    ------
    ValueError
        If normalization is not one of allowed values
    """

    values = values[np.isfinite(values)]

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
        msg = f"Invalid normalization {normalization}."
        raise ValueError(msg)

    return counts, bins


def create_bins(values: npt.NDArray[Any], bin_count: int) -> npt.NDArray[Any]:
    """
    Create evenly distributed bins for data.

    Creates evenly distributed bins while accounting for data without variance. Integer
    valued bins are created for integer data.

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
    if min_value == max_value:
        min_value -= 1
        max_value += 1
    bin_size = (max_value - min_value) / bin_count
    if np.issubdtype(values.dtype, np.integer):
        bin_size = max(round(bin_size), 1)
    bins = np.arange(min_value, max_value + bin_size, bin_size, dtype=values.dtype)
    bins = np.round(bins, 5)

    return bins


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


def ecdf(
    values: npt.NDArray[Any],
    *,
    name: str | None = None,
    color: str | None = None,
    legendgroup: str | None = None,
) -> go.Scatter | tuple[go.Scatter, go.Scatter]:
    """
    Create an empirical CDF plot.

    Creates a plot which shows the empirical cumulative distribution function. If given
    a data array that has more than one dimension, the empirical CDF of the flattened
    array is plotted along with the range of possible ecdf values. The range is
    calculated by flattening all but the last dimension of values and calculating the
    ecdfs of the rows of the resulting array.

    Parameters
    ----------
    values : numpy.ndarray of Any
        Values for the ecdf. If values has more than one dimension, the last dimension
        is interpreted as indexing different datasets.
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

    conditional_mean = np.sort(values.flatten())
    conditional_mean = conditional_mean[np.isfinite(conditional_mean)]
    if conditional_mean.size > _TOO_MANY_SCATTER:
        conditional_mean = conditional_mean[
            :: conditional_mean.size // _TOO_MANY_SCATTER
        ]
    cdf = go.Scatter(
        x=conditional_mean,
        y=np.linspace(0, 1, conditional_mean.size),
        mode="lines",
        name=name,
        line_color=color,
        hovertemplate="Arvo: %{x:.2f}<br>Kumulatiivinen yleisyys: %{y:.2f}"
        "<extra></extra>",
        legendgroup=legendgroup,
        legendgrouptitle_text=legendgroup,
    )

    if len(values.shape) > 1:
        values = np.sort(values.reshape(-1, values.shape[-1]), axis=-1)
        lower_bound = np.nanmin(values, 0)
        upper_bound = np.nanmax(values, 0)
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


def uniform_variation(
    sample_size: int, bin_count: int
) -> tuple[go.Scatter, go.Scatter]:
    """
    Visualize 99 % variation intervals for uniformly distributed data.

    Creates intervals for the expected variance of the histogram bin counts and
    empirical cumulative distribution function values calculated from uniformly
    distributed data.

    Since the data is uniformly distributed, each bin is equally likely and the
    probability of a sample falling into a bin is 1/bin_count. The distribution of
    repeated yes-no questions is binomial.

    The values of inverse empirical cumulative distribution are order statistics of
    uniform distribution. The distribution of the kth largest value in a sample is
    beta-distributed.

    Parameters
    ----------
    sample_size : int
        Sample size
    bin_count : int
        Number of bins in the histogram

    Returns
    -------
    go.Scatter
        Variation interval for histgram
    go.Scatter
        Variation interval for ecdf

    See Also
    --------
    https://en.wikipedia.org/wiki/Order_statistic#Probability_distributions_of_order_statistics
    For more in depth explanation about the distribution of ecdf values.
    """

    # Bin counts are binomially distributed
    bar_dist = binom(sample_size, 1 / bin_count)
    bar_bounds = [
        bar_dist.ppf(0.005) / sample_size * 100,
        bar_dist.ppf(0.995) / sample_size * 100,
    ]
    histogram_variation = go.Scatter(
        x=[0, 1, 1, 0],
        y=[bar_bounds[0], bar_bounds[0], bar_bounds[1], bar_bounds[1]],
        fill="toself",
        fillcolor="rgba(0,0,0,0.2)",
        line_color="rgba(0,0,0,0)",
        hoverinfo="skip",
    )

    # The kth order statistic (basically the inverse CDF) of a uniform variable on [0,1]
    # is beta-distributed
    lower_bound = np.linspace(0, 1, 1000)
    higher_bound = lower_bound.copy()
    left_bound = []
    right_bound = []
    for percentile_index in range(lower_bound.size):
        rank = percentile_index / lower_bound.size * sample_size
        cdf_dist = beta(rank, sample_size + 1 - rank)
        left_bound.append(cdf_dist.ppf(0.005))
        right_bound.append(cdf_dist.ppf(0.995))
        higher_bound[percentile_index] -= left_bound[-1]
        lower_bound[percentile_index] -= right_bound[-1]
    cdf_variation = go.Scatter(
        x=left_bound + right_bound[::-1],
        y=np.concatenate((higher_bound, lower_bound[::-1])),
        fill="toself",
        fillcolor="rgba(0,0,0,0.2)",
        line_color="rgba(0,0,0,0)",
        hoverinfo="skip",
    )

    return histogram_variation, cdf_variation


def hex_to_rgb(hex_value: str) -> tuple[str, str, str]:
    """
    Convert hex valued color to a RGB representation.

    Parameters
    ----------
    hex_value : str
        Color in hexadecimal format

    Returns
    -------
    tuple of (str, str, str)
        Red, blue and green components of the color
    """

    hex_value = hex_value.strip("#")

    return tuple(str(int(hex_value[i : i + 2], 16)) for i in (0, 2, 4))
