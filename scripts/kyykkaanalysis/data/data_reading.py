"""
Reading play times for CSV files.

This module provides functionalities for reading kyykkä play time data from CSV files
formatted using the format described in this kyykkaanalysis' README.

See Also
--------
README.md
"""

from pathlib import Path

import numpy as np
import structlog

from .data_classes import Game, Half, Konatime, Stream, Throwtime

_LOG = structlog.get_logger(__name__)


def read_times(input_file: Path, team_file: Path) -> list[Stream]:
    """
    Read play times from a CSV file.

    Reads, stores and validates timestamps of throws and kona completions from a CSV
    file.

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

    log = _LOG.bind(input_file=input_file, team_file=team_file)
    log.info("Reading kyykkä play time data.")

    if not input_file.exists():
        msg = f"Input file {input_file} does not exist."
        raise ValueError(msg)
    teams = _read_teams(team_file)

    log.info("Reading timestamps for events of kyykkä games.")
    player_ids = {}
    data = []
    with input_file.open(encoding="utf-8") as file:
        for i, line in enumerate(file):
            content = line.strip().split(",")
            if i % 3 == 0:
                log = log.bind(stream_index=i // 3)
                log.debug("Reading stream description.", row=i)
                url = content[0]
                pitch = content[1]
                if pitch == "":
                    pitch = "Kenttä 1"
                stream = Stream(url, pitch)
            elif i % 3 == 1:
                log.debug("Reading timestamps.", row=i)
                times = content[: _last_valid_time(content) + 1]
            else:
                log.debug(
                    "Reading players or events and matching them to timestamps.", row=i
                )
                players = content[: _last_valid_time(content) + 1]
                _read_stream_times(
                    teams,
                    player_ids,
                    stream,
                    times,
                    players,
                    playoffs=len(data) >= 13,  # noqa: PLR2004
                )
                log.debug(
                    "Read data from one stream.",
                    url=stream.url,
                    pitch=stream.pitch,
                    game_count=len(stream.games),
                )
                data.append(stream)

    log.info("Kyykkä play time data read.", stream_count=len(data))

    return data


def _read_teams(team_file: Path) -> dict[str, str]:
    _LOG.info("Reading teams of players.", team_file=team_file)
    if not team_file.exists():
        msg = "Input file does not exist."
        raise ValueError(msg)

    teams = {}
    with team_file.open(encoding="utf-8") as file:
        for line in file:
            player, team = line.strip().split(",")
            teams[player] = team

    _LOG.info(
        "Read teams of players.",
        team_file=team_file,
        player_count=len(player),
        team_count=len(set(teams.values())),
    )

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
    halfs = [Half()]
    konas = []
    for time_string, player in zip(times, players, strict=True):
        if player not in ["Kona kasassa", ""]:
            if len(player_ids) == 0:
                player_ids[player] = 0
            elif player not in player_ids:
                player_ids[player] = max(player_ids.values()) + 1

        if time_string == "?":
            time = np.datetime64("NaT")
        elif time_string == player == "":
            continue
        else:
            time = _parse_time(time_string)

        if player == "Kona kasassa":
            halfs, konas = _parse_kona_time(stream, halfs, konas, time)
        else:
            _validate_time(stream, halfs, time)

            halfs[-1].throws.append(
                Throwtime(player_ids[player], player, time, teams[player], playoffs)
            )

    halfs[-1].konas = (
        Konatime(np.datetime64("NaT")),
        Konatime(np.datetime64("NaT")),
    )
    stream.games.append(Game(tuple(halfs)))


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


def _validate_time(stream: Stream, halfs: list[Half], time: np.datetime64) -> None:
    if len(halfs) == 1:
        if len(halfs[-1].throws) == 0 and len(stream.games) == 0:
            previous_time = np.datetime64("NaT")
        elif len(halfs[-1].throws) == 0:
            previous_time = stream.end
        else:
            previous_time = halfs[-1].throws[-1].time
    elif len(halfs[-1].throws) == 0:
        previous_time = halfs[0].konas[-1].time
    else:
        previous_time = halfs[-1].throws[-1].time
    if time < previous_time:
        msg = (
            f"Throw timestamp {time} in stream {stream.url} is earlier than"
            f" the previous timestamp {previous_time}."
        )
        ValueError(msg)


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
