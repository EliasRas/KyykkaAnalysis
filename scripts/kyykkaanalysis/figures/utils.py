"""Utility functions for plotly figures"""
from pathlib import Path
from time import sleep

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
