"""Utility functions for plotly figures"""
from pathlib import Path
from time import sleep

from plotly import graph_objects as go


def write_pdf(figure: go.Figure, figure_path: Path) -> None:
    """
    Saving the file twice and waiting in between to fix https://github.com/plotly/plotly.py/issues/3469.
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
