"""Reading play times for CSV files"""
from pathlib import Path

import numpy as np

from .data_classes import Stream, Game, Half, Konatime, Throwtime


def read_times(input_file: Path) -> list[Stream]:
    """
    Read play times from a CSV file

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the file which contains the play times

    Returns
    -------
    list[Stream]
        _description_

    Raises
    ------
    ValueError
        If the data file does not exist
    ValueError
        If the data file contains a timestamp in invalid format
    """

    if not input_file.exists():
        raise ValueError("Input file does not exist.")

    player_ids = {}
    data = []
    with open(input_file, encoding="utf-8") as file:
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
                _read_stream_times(player_ids, stream, times, players)
                data.append(stream)

    return data


def _last_valid_time(content: list[str]) -> int:
    last_valid_index = len(content) - 1
    while content[last_valid_index] == "":
        last_valid_index -= 1

    return last_valid_index


def _read_stream_times(
    player_ids: dict[str, int], stream: Stream, times: list[str], players: list[str]
) -> None:
    halves = [Half()]
    konas = []
    for time, player in zip(times, players, strict=True):
        if len(player_ids) == 0:
            player_ids[player] = 1
        elif player not in player_ids:
            player_ids[player] = max(player_ids.values()) + 1

        if time == "?":
            time = np.datetime64("NaT")
        elif time == player == "":
            continue
        else:
            time = _parse_time(time)

        if player == "Kona kasassa":
            halves, konas = _parse_kona_time(stream, halves, konas, time)
        else:
            halves[-1].throws.append(Throwtime(player_ids[player], player, time))

    halves[-1].konas = (Konatime(np.datetime64("NaT")), Konatime(np.datetime64("NaT")))
    stream.games.append(Game(tuple(halves)))


def _parse_time(time_string: str) -> np.datetime64:
    time_info = time_string.split(".")
    if len(time_info) == 2:
        hours = np.timedelta64(0, "h")
        minutes = np.timedelta64(int(time_info[0]), "m")
        seconds = np.timedelta64(int(time_info[1]), "s")
    elif len(time_info) == 3:
        hours = np.timedelta64(int(time_info[0]), "h")
        minutes = np.timedelta64(int(time_info[1]), "m")
        seconds = np.timedelta64(int(time_info[2]), "s")
    else:
        raise ValueError("Invalid time format")

    return np.datetime64("2000-01-01") + hours + minutes + seconds


def _parse_kona_time(
    stream: Stream,
    halves: list[Half],
    konas: list[Konatime],
    time: np.datetime64,
) -> list[Konatime]:
    if len(konas) == 0:
        konas.append(Konatime(time))
    else:
        konas.append(Konatime(time))
        halves[-1].konas = tuple(konas)
        konas = []
        if len(halves) == 2:
            stream.games.append(Game(tuple(halves)))
            halves = []
        else:
            halves.append(Half())

    return konas
