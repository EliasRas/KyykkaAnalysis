"""Visualizations for data"""
from pathlib import Path

import numpy as np
from numpy import typing as npt
from plotly import graph_objects as go, colors

from .utils import write_pdf
from ..data.data_classes import Stream

PLOT_COLORS = colors.qualitative.Plotly
FONT_SIZE_2X2 = 28
FONT_SIZE_BOXPLOT = 10
FONT_SIZE = 15


def timeline(data: list[Stream]):
    """
    Plot the timeline of the throws in a game

    Parameters
    ----------
    data : list of Stream
        Play time data
    """

    for stream in data:
        figure = go.Figure()
        index_start = 0
        for game_index, game in enumerate(stream.games):
            for half_index, half in enumerate(game.halfs):
                players = half.players()
                throw_times = half.throw_times()
                if np.isfinite(throw_times).sum() > 0:
                    time_indices = np.arange(throw_times.size) + index_start
                    figure.add_trace(
                        go.Scatter(
                            x=time_indices,
                            y=throw_times,
                            customdata=players,
                            mode="markers",
                            marker_color=PLOT_COLORS[game_index % len(PLOT_COLORS)],
                            hovertemplate=f"{game_index+1}. pelin {half_index+1}. erä<br>"
                            "Heittäjä: %{customdata}<br>Heittoaika: %{y|%H.%M.%S}"
                            "<extra></extra>",
                            showlegend=False,
                        )
                    )
                    index_start += throw_times.size

                    time_differences = np.diff(throw_times)
                    invalid_times = time_differences <= np.timedelta64(0, "s")
                    figure.add_trace(
                        go.Scatter(
                            x=time_indices[1:][invalid_times],
                            y=throw_times[1:][invalid_times],
                            mode="markers",
                            marker={"color": "black", "symbol": "circle-open"},
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                kona_times = np.array([kona.time for kona in half.konas])
                if np.isfinite(kona_times).sum() > 0:
                    figure.add_trace(
                        go.Scatter(
                            x=np.arange(kona_times.size) + index_start,
                            y=kona_times,
                            mode="markers",
                            marker_symbol="x",
                            marker_color=PLOT_COLORS[game_index % len(PLOT_COLORS)],
                            hovertemplate=f"{game_index+1}. pelin {half_index+1}. erä<br>"
                            "Kasausaika: %{y|%H.%M.%S}"
                            "<extra></extra>",
                            showlegend=False,
                        )
                    )

                    index_start += kona_times.size

        figure.update_layout(
            title=f"{stream.url}<br>{stream.pitch}",
            title_x=0.5,
            yaxis={"title": "Tapahtuma-aika", "tickformat": "%H.%M.%S"},
            xaxis_title="Järjestysnumero",
            separators=", ",
        )
        figure.show()


def time_distributions(data: list[Stream], figure_directory: Path):
    """
    Plot the distributions of the various durations

    Parameters
    ----------
    data : list of Stream
        Play time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    _throw_distributions(data, figure_directory)
    _kona_distribution(data, figure_directory)
    _game_distributions(data, figure_directory)


def _throw_distributions(data: list[Stream], figure_directory: Path) -> None:
    throw_times: npt.NDArray[np.timedelta64] = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )

    _game_throws(data, throw_times, figure_directory)
    _position_throws(data, throw_times, figure_directory)
    _order_throws1(data, throw_times, figure_directory)
    _order_throws2(data, throw_times, figure_directory)


def _game_throws(
    data: list[Stream], throw_times: npt.NDArray[np.timedelta64], figure_directory: Path
) -> None:
    playoffs = np.concatenate([stream.playoffs(difference=True) for stream in data])

    figure = go.Figure(
        go.Histogram(
            x=throw_times[np.isfinite(throw_times) & playoffs].astype(int),
            nbinsx=50,
            histnorm="probability density",
            opacity=0.5,
            marker_color=PLOT_COLORS[0],
            name="Runkosarjapelit",
            hovertemplate="Heittojen välinen aika: %{x} s<br>Suhteellinen yleisyys: %{y:.3f}",
        )
    )
    figure.add_trace(
        go.Histogram(
            x=throw_times[np.isfinite(throw_times) & ~playoffs].astype(int),
            nbinsx=50,
            histnorm="probability density",
            opacity=0.5,
            marker_color=PLOT_COLORS[1],
            name="Pudotuspelit",
            hovertemplate="Heittojen välinen aika: %{x} s<br>Suhteellinen yleisyys: %{y:.3f}",
        )
    )

    figure.update_layout(
        barmode="overlay",
        xaxis_title="Heittojen välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "heitot1.html")
    write_pdf(figure, figure_directory / "heitot1.pdf")


def _position_throws(
    data: list[Stream], throw_times: npt.NDArray[np.timedelta64], figure_directory: Path
) -> None:
    player_positions = np.concatenate(
        [stream.positions(difference=True) for stream in data]
    )

    figure = go.Figure()
    for position in range(1, 5):
        figure.add_trace(
            go.Histogram(
                x=throw_times[
                    np.isfinite(throw_times) & (player_positions == position)
                ].astype(int),
                nbinsx=50,
                histnorm="probability density",
                opacity=0.5,
                marker_color=PLOT_COLORS[position - 1],
                name=f"{position}. heittäjä",
                hovertemplate="Heittojen välinen aika: %{x} s<br>"
                "Suhteellinen yleisyys: %{y:.3f}",
            ),
        )

    figure.update_layout(
        barmode="overlay",
        xaxis_title="Heittojen välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "heitot2.html")
    write_pdf(figure, figure_directory / "heitot2.pdf")


def _order_throws1(
    data: list[Stream], throw_times: npt.NDArray[np.timedelta64], figure_directory: Path
) -> None:
    throw_numbers = np.concatenate(
        [stream.throw_numbers(difference=True) for stream in data]
    )

    figure = go.Figure()
    for throw in range(1, 5):
        figure.add_trace(
            go.Histogram(
                x=throw_times[
                    np.isfinite(throw_times) & (throw_numbers == throw)
                ].astype(int),
                nbinsx=50,
                histnorm="probability density",
                opacity=0.5,
                marker_color=PLOT_COLORS[throw - 1],
                name=f"{throw}. heitto",
                hovertemplate="Heittojen välinen aika: %{x} s<br>"
                "Suhteellinen yleisyys: %{y:.3f}",
            ),
        )

    figure.update_layout(
        barmode="overlay",
        xaxis_title="Heittojen välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "heitot3.html")
    write_pdf(figure, figure_directory / "heitot3.pdf")


def _order_throws2(
    data: list[Stream], throw_times: npt.NDArray[np.timedelta64], figure_directory: Path
) -> None:
    throw_numbers = (
        np.remainder(
            np.concatenate([stream.throw_numbers(difference=True) for stream in data])
            - 1,
            2,
        )
        + 1
    )

    figure = go.Figure()
    for throw in range(1, 3):
        figure.add_trace(
            go.Histogram(
                x=throw_times[
                    np.isfinite(throw_times) & (throw_numbers == throw)
                ].astype(int),
                nbinsx=50,
                histnorm="probability density",
                opacity=0.5,
                marker_color=PLOT_COLORS[throw - 1],
                name=f"{throw}. heitto",
                hovertemplate="Heittojen välinen aika: %{x} s<br>"
                "Suhteellinen yleisyys: %{y:.3f}",
            )
        )

    figure.update_layout(
        barmode="overlay",
        xaxis_title="Heittojen välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "heitot4.html")
    write_pdf(figure, figure_directory / "heitot4.pdf")


def _kona_distribution(data: list[Stream], figure_directory: Path) -> None:
    kona_times = np.concatenate([stream.kona_times(difference=True) for stream in data])
    kona_times = kona_times[np.isfinite(kona_times)]
    figure = go.Figure(
        go.Histogram(
            x=kona_times.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Kasausaika: %{x} s<br>Kasausten määrä: %{y}",
            showlegend=False,
        )
    )
    figure.update_layout(
        xaxis_title="Konan kasaukseen käytetty aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
    )
    figure.write_html(figure_directory / "konat.html")
    write_pdf(figure, figure_directory / "konat.pdf")


def _game_distributions(data: list[Stream], figure_directory: Path) -> None:
    _half_duration(data, figure_directory)
    _game_duration(data, figure_directory)
    _half_break(data, figure_directory)
    _game_break(data, figure_directory)


def _half_duration(data: list[Stream], figure_directory: Path) -> None:
    half_durations = np.concatenate([stream.half_durations for stream in data])
    half_durations = half_durations[np.isfinite(half_durations)]
    figure = go.Figure(
        go.Histogram(
            x=half_durations.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Erän kesto: %{x} s<br>Erien määrä: %{y}",
            showlegend=False,
        )
    )
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Erien kesto [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    write_pdf(figure, figure_directory / "pelit1.pdf")


def _game_duration(data: list[Stream], figure_directory: Path) -> None:
    game_durations = np.concatenate([stream.game_durations for stream in data])
    game_durations = game_durations[np.isfinite(game_durations)]
    figure = go.Figure(
        go.Histogram(
            x=game_durations.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Pelin kesto: %{x} s<br>Pelien määrä: %{y}",
            showlegend=False,
        )
    )
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Pelin kesto [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "pelit2.html")
    write_pdf(figure, figure_directory / "pelit2.pdf")


def _half_break(data: list[Stream], figure_directory: Path) -> None:
    half_breaks = np.concatenate([stream.half_breaks for stream in data])
    half_breaks = half_breaks[np.isfinite(half_breaks)]
    figure = go.Figure(
        go.Histogram(
            x=half_breaks.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Erien välisen tauon kesto: %{x} s<br>Taukojen määrä: %{y}",
            showlegend=False,
        )
    )
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Erien välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "pelit3.html")
    write_pdf(figure, figure_directory / "pelit3.pdf")


def _game_break(data: list[Stream], figure_directory: Path) -> None:
    game_breaks = np.concatenate([stream.game_breaks for stream in data])
    game_breaks = game_breaks[np.isfinite(game_breaks)]
    figure = go.Figure(
        go.Histogram(
            x=game_breaks.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Pelien välisen tauon kesto: %{x} s<br>Taukojen määrä: %{y}",
            showlegend=False,
        ),
    )
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Pelien välinen aika [s]",
        yaxis_showticklabels=False,
        separators=", ",
        font={"size": FONT_SIZE_2X2, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "pelit4.html")
    write_pdf(figure, figure_directory / "pelit4.pdf")


def averages(data: list[Stream], figure_directory: Path):
    """
    Plot the distribution of the average throw times of players

    Parameters
    ----------
    data : list of Stream
        Play time data
    figure_directory : Path
        Path to the directory in which the figure is saved
    """

    _player_averages(data, figure_directory)
    _team_averages(data, figure_directory)


def _player_averages(data: list[Stream], figure_directory: Path) -> None:
    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    players = np.concatenate([stream.players(difference=True) for stream in data])
    throws = np.concatenate([stream.throw_numbers(difference=True) for stream in data])

    unique_players = np.unique(players)
    time_averages = []
    counts = []
    medians = []
    for player in unique_players:
        player_times = throw_times[(players == player) & np.isfinite(throw_times)]
        time_averages.append(np.mean(player_times.astype(int)))
        counts.append(player_times.size)
        medians.append(np.median(player_times.astype(int)))
    counts = np.array(counts).reshape(-1, 1)

    figure = go.Figure(
        go.Box(
            y=time_averages,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot",
            boxpoints="all",
            marker_color=PLOT_COLORS[0],
            hovertemplate="Heittäjä: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
        )
    )
    figure.add_trace(
        go.Box(
            y=medians,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot, mediaani",
            boxpoints="all",
            marker_color=PLOT_COLORS[1],
            hovertemplate="Heittäjä: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
        )
    )

    for throw in range(1, 5):
        time_averages = []
        counts = []
        for player in unique_players:
            player_times = throw_times[
                (players == player) & np.isfinite(throw_times) & (throws == throw)
            ]
            time_averages.append(np.nanmean(player_times.astype(int)))
            counts.append(player_times.size)
        counts = np.array(counts).reshape(-1, 1)

        figure.add_trace(
            go.Box(
                y=time_averages,
                customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
                name=f"{throw}. heitto",
                boxpoints="all",
                marker_color=PLOT_COLORS[throw + 1],
                hovertemplate="Heittäjä: %{customdata[0]}<br>"
                "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
                "Heittojen määrä: %{customdata[1]}",
            )
        )

    figure.update_layout(
        legend_groupclick="toggleitem",
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "keskiarvot1.html")
    figure.update_layout(showlegend=False)
    write_pdf(figure, figure_directory / "keskiarvot1.pdf")


def _team_averages(data: list[Stream], figure_directory: Path) -> None:
    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    teams = np.concatenate([stream.teams(difference=True) for stream in data])
    throws = np.concatenate([stream.throw_numbers(difference=True) for stream in data])

    unique_players = np.unique(teams)
    time_averages = []
    counts = []
    medians = []
    for player in unique_players:
        player_times = throw_times[(teams == player) & np.isfinite(throw_times)]
        time_averages.append(np.mean(player_times.astype(int)))
        counts.append(player_times.size)
        medians.append(np.median(player_times.astype(int)))
    counts = np.array(counts).reshape(-1, 1)

    figure = go.Figure(
        go.Box(
            y=time_averages,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot",
            boxpoints="all",
            marker_color=PLOT_COLORS[0],
            hovertemplate="Joukkue: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
        )
    )
    figure.add_trace(
        go.Box(
            y=medians,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot, mediaani",
            boxpoints="all",
            marker_color=PLOT_COLORS[1],
            hovertemplate="Joukkue: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
        )
    )

    for throw in range(1, 5):
        time_averages = []
        counts = []
        for player in unique_players:
            player_times = throw_times[
                (teams == player) & np.isfinite(throw_times) & (throws == throw)
            ]
            time_averages.append(np.nanmean(player_times.astype(int)))
            counts.append(player_times.size)
        counts = np.array(counts).reshape(-1, 1)

        figure.add_trace(
            go.Box(
                y=time_averages,
                customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
                name=f"{throw}. heitto",
                boxpoints="all",
                marker_color=PLOT_COLORS[throw + 1],
                hovertemplate="Joukkue: %{customdata[0]}<br>"
                "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
                "Heittojen määrä: %{customdata[1]}",
            )
        )

    figure.update_layout(
        legend_groupclick="toggleitem",
        separators=", ",
        font={"size": FONT_SIZE, "family": "Computer modern"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    figure.write_html(figure_directory / "keskiarvot2.html")
    figure.update_layout(showlegend=False)
    write_pdf(figure, figure_directory / "keskiarvot2.pdf")
