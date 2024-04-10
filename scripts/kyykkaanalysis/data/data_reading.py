"""Reading play times for CSV files."""

from pathlib import Path

import numpy as np

from .data_classes import Game, Half, Konatime, Stream, Throwtime


def read_times(input_file: Path, team_file: Path) -> list[Stream]:
    """
    Read play times from a CSV file.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the file which contains the play times
    team_file : pathlib.Path
        Path to the file which contains the teams for the players

    Returns
    -------
    list of Stream
        Play times

    Raises
    ------
    ValueError
        If any of the data files does not exist
    ValueError
        If the play time file contains a timestamp in invalid format
    ValueError
        If the play time file contains a timestamps in unchronological order
    """

    if not input_file.exists():
        msg = f"Input file {input_file} does not exist."
        raise ValueError(msg)
    teams = _read_teams(team_file)

    player_ids = {}
    data = []
    with input_file.open(encoding="utf-8") as file:
        for i, line in enumerate(file):
            content = line.strip().split(",")
            if i % 3 == 0:
                url = content[0]
                pitch = content[1]
                if pitch == "":
                    pitch = "Kenttä 1"
                stream = Stream(url, pitch)
            elif i % 3 == 1:
                times = content[: _last_valid_time(content) + 1]
            else:
                players = content[: _last_valid_time(content) + 1]
                _read_stream_times(
                    teams,
                    player_ids,
                    stream,
                    times,
                    players,
                    playoffs=len(data) >= 13,  # noqa: PLR2004
                )
                data.append(stream)

    return data


def _read_teams(team_file: Path) -> dict[str, str]:
    if not team_file.exists():
        msg = "Input file does not exist."
        raise ValueError(msg)

    teams = {}
    with team_file.open(encoding="utf-8") as file:
        for line in file:
            player, team = line.strip().split(",")
            teams[player] = team

    return teams


def _last_valid_time(content: list[str]) -> int:
    last_valid_index = len(content) - 1
    while content[last_valid_index] == "":
        last_valid_index -= 1

    return last_valid_index


def _read_stream_times(  # noqa: PLR0913
    teams: dict[str, str],
    player_ids: dict[str, int],
    stream: Stream,
    times: list[str],
    players: list[str],
    *,
    playoffs: bool,
) -> None:
    halves = [Half()]
    konas = []
    for time_string, player in zip(times, players, strict=True):
        if player not in ["Kona kasassa", ""]:
            if len(player_ids) == 0:
                player_ids[player] = 0
            elif player not in player_ids:
                player_ids[player] = max(player_ids.values()) + 1

        if time_string == "?":
            time = np.datetime64("NaT")
        elif time == player == "":
            continue
        else:
            time = _parse_time(time_string)

        if player == "Kona kasassa":
            halves, konas = _parse_kona_time(stream, halves, konas, time)
        else:
            if len(halves) == 1:
                if len(halves[-1].throws) == 0 and len(stream.games) == 0:
                    previous_time = np.datetime64("NaT")
                elif len(halves[-1].throws) == 0:
                    previous_time = stream.end
                else:
                    previous_time = halves[-1].throws[-1].time
            elif len(halves[-1].throws) == 0:
                previous_time = halves[0].konas[-1].time
            else:
                previous_time = halves[-1].throws[-1].time
            if time < previous_time:
                msg = (
                    f"Throw timestamp {time} in stream {stream.url} is earlier than"
                    f" the previous timestamp {previous_time}."
                )
                ValueError(msg)

            halves[-1].throws.append(
                Throwtime(player_ids[player], player, time, teams[player], playoffs)
            )

    halves[-1].konas = (
        Konatime(np.datetime64("NaT")),
        Konatime(np.datetime64("NaT")),
    )
    stream.games.append(Game(tuple(halves)))


def _parse_time(time_string: str) -> np.datetime64:
    time_info = time_string.split(".")
    if len(time_info) == 2:  # noqa: PLR2004
        hours = np.timedelta64(0, "h")
        minutes = np.timedelta64(int(time_info[0]), "m")
        seconds = np.timedelta64(int(time_info[1]), "s")
    elif len(time_info) == 3:  # noqa: PLR2004
        hours = np.timedelta64(int(time_info[0]), "h")
        minutes = np.timedelta64(int(time_info[1]), "m")
        seconds = np.timedelta64(int(time_info[2]), "s")
    else:
        msg = "Invalid time format"
        raise ValueError(msg)

    return np.datetime64("2000-01-01") + hours + minutes + seconds


def _parse_kona_time(
    stream: Stream,
    halves: list[Half],
    konas: list[Konatime],
    time: np.datetime64,
) -> tuple[list[Half], list[Konatime]]:
    previous_time = halves[-1].throws[-1].time if len(konas) == 0 else konas[-1].time
    if time < previous_time:
        msg = (
            f"Kona completion timestamp {time} in stream {stream.url} is earlier than"
            f" the previous timestamp {previous_time}."
        )
        ValueError(msg)

    if len(konas) == 0:
        konas.append(Konatime(time))
    else:
        konas.append(Konatime(time))
        halves[-1].konas = tuple(konas)
        konas = []
        if len(halves) == 2:  # noqa: PLR2004
            stream.games.append(Game(tuple(halves)))
            halves = [Half()]
        else:
            halves.append(Half())

    return halves, konas
