"""Visualizations for data"""

import numpy as np
from plotly import graph_objects as go, colors
from plotly.subplots import make_subplots

from ..data.data_classes import Stream

PLOT_COLORS = colors.qualitative.Plotly


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


def time_distributions(data: list[Stream]):
    """
    Plot the distributions of the various durations

    Parameters
    ----------
    data : list of Stream
        Play time data
    """

    _throw_distributions(data)
    _game_distributions(data)


def _throw_distributions(data: list[Stream]) -> None:
    figure = make_subplots(rows=2, cols=2)

    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    playoffs = np.concatenate([stream.playoffs(difference=True) for stream in data])
    figure.add_trace(
        go.Histogram(
            x=throw_times[np.isfinite(throw_times) & playoffs].astype(int),
            nbinsx=50,
            histnorm="probability density",
            opacity=0.5,
            name="Runkosarjapelit",
            hovertemplate="Heittojen välinen aika: %{x} s<br>Suhteellinen yleisyys: %{y}",
            legendgroup="Pelit",
            legendgrouptitle_text="Pelit",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=throw_times[np.isfinite(throw_times) & ~playoffs].astype(int),
            nbinsx=50,
            histnorm="probability density",
            opacity=0.5,
            name="Playoffspelit",
            hovertemplate="Heittojen välinen aika: %{x} s<br>Suhteellinen yleisyys: %{y}",
            legendgroup="Pelit",
            legendgrouptitle_text="Pelit",
        ),
        row=1,
        col=1,
    )

    player_positions = np.concatenate(
        [stream.positions(difference=True) for stream in data]
    )
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
                legendgroup="Heittopaikka",
                legendgrouptitle_text="Heittopaikoittain",
            ),
            row=1,
            col=2,
        )

    throw_numbers = np.concatenate(
        [stream.throw_numbers(difference=True) for stream in data]
    )
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
                legendgroup="Heitto",
                legendgrouptitle_text="Heitoittain",
            ),
            row=2,
            col=1,
        )

    kona_times = np.concatenate([stream.kona_times(difference=True) for stream in data])
    kona_times = kona_times[np.isfinite(kona_times)]
    figure.add_trace(
        go.Histogram(
            x=kona_times.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Kasausaika: %{x} s<br>Kasausten määrä: %{y}",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Heittojen välinen aika [s]", row=1, col=1)
    figure.update_xaxes(title_text="Heittojen välinen aika [s]", row=1, col=2)
    figure.update_xaxes(title_text="Heittojen välinen aika [s]", row=2, col=1)
    figure.update_xaxes(title_text="Konan kasaukseen käytetty aika [s]", row=2, col=2)
    figure.update_layout(
        barmode="overlay",
        legend_groupclick="toggleitem",
        separators=", ",
    )
    figure.show()


def _game_distributions(data: list[Stream]) -> None:
    figure = make_subplots(rows=2, cols=2)
    half_durations = np.concatenate([stream.half_durations for stream in data])
    half_durations = half_durations[np.isfinite(half_durations)]
    figure.add_trace(
        go.Histogram(
            x=half_durations.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Erän kesto: %{x} s<br>Erien määrä: %{y}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    game_durations = np.concatenate([stream.game_durations for stream in data])
    game_durations = game_durations[np.isfinite(game_durations)]
    figure.add_trace(
        go.Histogram(
            x=game_durations.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Pelin kesto: %{x} s<br>Pelien määrä: %{y}",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    half_breaks = np.concatenate([stream.half_breaks for stream in data])
    half_breaks = half_breaks[np.isfinite(half_breaks)]
    figure.add_trace(
        go.Histogram(
            x=half_breaks.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Erien välisen tauon kesto: %{x} s<br>Taukojen määrä: %{y}",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    game_breaks = np.concatenate([stream.game_breaks for stream in data])
    game_breaks = game_breaks[np.isfinite(game_breaks)]
    figure.add_trace(
        go.Histogram(
            x=game_breaks.astype(int),
            nbinsx=25,
            marker_color=PLOT_COLORS[0],
            hovertemplate="Pelien välisen tauon kesto: %{x} s<br>Taukojen määrä: %{y}",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Erien kesto [s]", row=1, col=1)
    figure.update_xaxes(title_text="Pelin kesto [s]", row=1, col=2)
    figure.update_xaxes(title_text="Erien välinen aika [s]", row=2, col=1)
    figure.update_xaxes(title_text="Pelien välinen aika [s]", row=2, col=2)
    figure.update_layout(
        barmode="overlay",
        legend_groupclick="toggleitem",
        separators=", ",
    )
    figure.show()


def averages(data: list[Stream]):
    """
    Plot the distribution of the average throw times of players

    Parameters
    ----------
    data : list of Stream
        Play time data
    """

    figure = make_subplots(rows=2, cols=1, subplot_titles=["Pelaajat", "Joukkueet"])
    _player_averages(data, figure)
    _team_averages(data, figure)
    figure.show()


def _player_averages(data: list[Stream], figure: go.Figure) -> None:
    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    players = np.concatenate([stream.players(difference=True) for stream in data])
    throws = np.concatenate([stream.throw_numbers(difference=True) for stream in data])

    unique_players = np.unique(players)
    averages = []
    counts = []
    medians = []
    for player in unique_players:
        player_times = throw_times[(players == player) & np.isfinite(throw_times)]
        averages.append(np.mean(player_times.astype(int)))
        counts.append(player_times.size)
        medians.append(np.median(player_times.astype(int)))
    counts = np.array(counts).reshape(-1, 1)

    figure.add_trace(
        go.Box(
            y=averages,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot",
            boxpoints="all",
            hovertemplate="Heittäjä: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
            legendgroup="players",
            legendgrouptitle_text="Pelaajat",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Box(
            y=medians,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot, mediaani",
            boxpoints="all",
            hovertemplate="Heittäjä: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
            legendgroup="players",
            legendgrouptitle_text="Pelaajat",
        ),
        row=1,
        col=1,
    )

    for throw in range(1, 5):
        averages = []
        counts = []
        for player in unique_players:
            player_times = throw_times[
                (players == player) & np.isfinite(throw_times) & (throws == throw)
            ]
            averages.append(np.nanmean(player_times.astype(int)))
            counts.append(player_times.size)
        counts = np.array(counts).reshape(-1, 1)

        figure.add_trace(
            go.Box(
                y=averages,
                customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
                name=f"{throw}. heitto",
                boxpoints="all",
                hovertemplate="Heittäjä: %{customdata[0]}<br>"
                "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
                "Heittojen määrä: %{customdata[1]}",
                legendgroup="players",
                legendgrouptitle_text="Pelaajat",
            ),
            row=1,
            col=1,
        )


def _team_averages(data: list[Stream], figure: go.Figure) -> None:
    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    teams = np.concatenate([stream.teams(difference=True) for stream in data])
    throws = np.concatenate([stream.throw_numbers(difference=True) for stream in data])

    unique_players = np.unique(teams)
    averages = []
    counts = []
    medians = []
    for player in unique_players:
        player_times = throw_times[(teams == player) & np.isfinite(throw_times)]
        averages.append(np.mean(player_times.astype(int)))
        counts.append(player_times.size)
        medians.append(np.median(player_times.astype(int)))
    counts = np.array(counts).reshape(-1, 1)

    figure.add_trace(
        go.Box(
            y=averages,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot",
            boxpoints="all",
            hovertemplate="Joukkue: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
            legendgroup="teams",
            legendgrouptitle_text="Joukkueet",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Box(
            y=medians,
            customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
            name="Kaikki heitot, mediaani",
            boxpoints="all",
            hovertemplate="Joukkue: %{customdata[0]}<br>"
            "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
            "Heittojen määrä: %{customdata[1]}",
            legendgroup="teams",
            legendgrouptitle_text="Joukkueet",
        ),
        row=2,
        col=1,
    )

    for throw in range(1, 5):
        averages = []
        counts = []
        for player in unique_players:
            player_times = throw_times[
                (teams == player) & np.isfinite(throw_times) & (throws == throw)
            ]
            averages.append(np.nanmean(player_times.astype(int)))
            counts.append(player_times.size)
        counts = np.array(counts).reshape(-1, 1)

        figure.add_trace(
            go.Box(
                y=averages,
                customdata=np.hstack((unique_players.reshape(-1, 1), counts)),
                name=f"{throw}. heitto",
                boxpoints="all",
                hovertemplate="Joukkue: %{customdata[0]}<br>"
                "Heiton keskimääräinen kesto: %{y:.1f} s<br>"
                "Heittojen määrä: %{customdata[1]}",
                legendgroup="teams",
                legendgrouptitle_text="Joukkueet",
            ),
            row=2,
            col=1,
        )
