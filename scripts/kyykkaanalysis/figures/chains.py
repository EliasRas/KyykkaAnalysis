"""
Visualizations about MCMC chains.

This module provides plotting functions for analyzing MCMC chains and their performance.
"""

from pathlib import Path

import numpy as np
import structlog
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from xarray import Dataset

from .utils import (
    FONT_SIZE,
    PLOT_COLORS,
    parameter_to_latex,
    precalculated_histogram,
)

_LOG = structlog.get_logger(__name__)


def chain_plots(
    samples: Dataset,
    figure_directory: Path,
    *,
    sample_stats: Dataset = None,
) -> None:
    """
    Plot information about MCMC chains.

    This function creates trace plots of MCMC chains. It also creates plots of NUTS
    divergences and the distributions of energy transitions and marginal energies if
    sampler statistics are provided.

    Parameters
    ----------
    samples : xarray.Dataset
        Posterior samples
    figure_directory : Path
        Path to the directory in which the figures are saved
    sample_stats : xarray.Dataset, optional
        Information about posterior samples
    """

    log = _LOG.bind(
        figure_directory=figure_directory, stats_exist=sample_stats is not None
    )
    log.info("Creating chain figures.")

    figure_directory.mkdir(parents=True, exist_ok=True)

    _trace_plot(samples, figure_directory)
    if sample_stats is not None:
        _divergences(samples, sample_stats, figure_directory)
        _energy_plot(sample_stats, figure_directory)

    log.info("Chain figures created.")


def _trace_plot(samples: Dataset, figure_directory: Path) -> None:
    samples = samples.drop_vars(["theta", "eta"], errors="ignore")
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

    _LOG.debug("Trace plot created.", figure_path=figure_directory / "chains.html")


def _divergences(
    samples: Dataset, sample_stats: Dataset, figure_directory: Path
) -> None:
    dimensions = []
    for parameter in sorted(samples.keys()):
        if parameter in ["theta", "eta"]:
            parameter_samples = samples[parameter].values
            parameter_samples = parameter_samples.reshape(
                -1, parameter_samples.shape[-1]
            )

            for player_index in range(parameter_samples.shape[-1]):
                player_samples = parameter_samples[:, player_index]
                min_value = player_samples.min()
                max_value = player_samples.max()
                margin = (max_value - min_value) * 0.1
                dimensions.append(
                    {
                        "range": [
                            np.floor(min_value - margin),
                            np.ceil(max_value + margin),
                        ],
                        "label": f"{parameter}_{player_index}",
                        "values": player_samples,
                        "tickvals": [],
                    }
                )
        else:
            parameter_samples = samples[parameter].values.flatten()

            min_value = parameter_samples.min()
            max_value = parameter_samples.max()
            margin = (max_value - min_value) * 0.1
            dimensions.append(
                {
                    "range": [
                        np.floor(min_value - margin),
                        np.ceil(max_value + margin),
                    ],
                    "label": parameter,
                    "values": parameter_samples,
                    "tickvals": [],
                }
            )

    divergences = sample_stats["diverging"].values.flatten().astype(int)
    if divergences.size > samples["draw"].size:
        sub_sampling_step = divergences.size // samples["draw"].size
        divergences = divergences[::sub_sampling_step]
    figure = go.Figure(
        go.Parcoords(
            line={
                "color": divergences,
                "colorscale": [(0, "black"), (1, "red")],
                "cmin": 0,
                "cmax": 1,
            },
            dimensions=dimensions,
        )
    )
    figure.update_layout(
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "divergences.html", include_mathjax="cdn")
    _LOG.debug(
        "Divergence plot created.", figure_path=figure_directory / "divergences.html"
    )


def _energy_plot(sample_stats: Dataset, figure_directory: Path) -> None:
    energies = sample_stats["energy"].values
    energy_transitions = np.diff(energies)

    chain_count = energies.shape[0]
    col_count = int(np.ceil(np.sqrt(chain_count)))
    row_count = int(np.ceil(chain_count / col_count))
    figure = make_subplots(
        rows=row_count,
        cols=col_count,
        subplot_titles=[f"Ketju {chain_index+1}" for chain_index in range(chain_count)],
    )
    for chain_index in range(chain_count):
        figure.add_traces(
            precalculated_histogram(
                energies[chain_index, :] - energies.mean(),
                PLOT_COLORS[0],
                name="Reunaenergia",
                normalization="probability density",
                legendgroup=f"Ketju {chain_index +1}",
            ),
            rows=chain_index // col_count + 1,
            cols=chain_index % col_count + 1,
        )
        figure.add_traces(
            precalculated_histogram(
                energy_transitions[chain_index, :],
                PLOT_COLORS[1],
                name="Energiasiirtymät",
                normalization="probability density",
                legendgroup=f"Ketju {chain_index +1}",
            ),
            rows=chain_index // col_count + 1,
            cols=chain_index % col_count + 1,
        )
    figure.update_layout(
        showlegend=False,
        barmode="overlay",
        bargap=0,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "energies.html", include_mathjax="cdn")
    _LOG.debug("Energy plot created.", figure_path=figure_directory / "energies.html")
