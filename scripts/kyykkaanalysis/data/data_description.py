"""Key numbers from the data."""

import numpy as np

from .data_classes import Stream


def print_description(data: list[Stream]) -> None:
    """
    Print a description of the data set.

    Parameters
    ----------
    data : list of Stream
        Play times
    """

    _count_description(data)
    _key_throw_values(data)
    _key_game_values(data)

    kona_times = np.concatenate([stream.kona_times(difference=True) for stream in data])
    kona_times = kona_times[np.isfinite(kona_times)]
    print(
        "Konan kasaamiseen käytetiin keskimäärin",
        _format_to_minutes(np.mean(kona_times.astype(int))),
        "sekuntia",
    )


def _count_description(data: list[Stream]) -> None:
    print("Kerätyssä datasetissä on")
    print("\t", len(data), "lähetystä")
    print(
        "\t",
        sum(
            len(stream.games)
            for stream in data
            if np.isfinite(stream.throw_times()).sum() > 0
        ),
        "peliä",
    )
    print(
        "\t",
        sum(
            len(game.halfs)
            for stream in data
            for game in stream.games
            if np.isfinite(game.throw_times()).sum() > 0
        ),
        "erää",
    )

    throw_times = np.concatenate([stream.throw_times() for stream in data])
    print("\t", np.isfinite(throw_times).sum(), "heittoa")
    throw_times2 = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    print("\t", np.isfinite(throw_times2).sum(), "heittoaikaa")
    kona_times = np.concatenate([stream.kona_times() for stream in data])
    print("\t", np.isfinite(kona_times).sum(), "konan kasausta")


def _key_throw_values(data: list[Stream]) -> None:
    throw_times = np.concatenate(
        [stream.throw_times(difference=True) for stream in data]
    )
    valid_times = np.isfinite(throw_times)
    throw_times = throw_times[valid_times]
    print(
        "Heittoaikojen keskiarvo:",
        round(throw_times.astype(int).mean(), 1),
        "sekuntia ja vaihteluväli ["
        f"{round(throw_times.astype(int).min(), 1)},"
        f" {round(throw_times.astype(int).max(), 1)}]",
    )

    throws = np.concatenate([stream.throw_numbers(difference=True) for stream in data])
    throws = throws[valid_times]
    first_throw = (throws == 1) | (throws == 3)
    print(
        "Ensimmäisen ja toisen heiton eron keskiarvo:",
        round(
            throw_times[first_throw].astype(int).mean()
            - throw_times[~first_throw].astype(int).mean(),
            1,
        ),
        "sekuntia",
    )


def _key_game_values(data: list[Stream]) -> None:
    half_durations = np.concatenate([stream.half_durations for stream in data])
    half_durations = half_durations[np.isfinite(half_durations)]
    print(
        "Erän keskimääräinen kesto",
        _format_to_minutes(half_durations.astype(int).mean()),
        "ja vaihteluväli ["
        f"{_format_to_minutes(half_durations.astype(int).min())}, "
        f"{_format_to_minutes(half_durations.astype(int).max())}"
        "]",
    )
    game_durations = np.concatenate([stream.game_durations for stream in data])
    game_durations = game_durations[np.isfinite(game_durations)]
    print(
        "Pelin keskimääräinen kesto",
        _format_to_minutes(game_durations.astype(int).mean()),
        "ja vaihteluväli ["
        f"{_format_to_minutes(game_durations.astype(int).min())}, "
        f"{_format_to_minutes(game_durations.astype(int).max())}"
        "]",
    )
    half_breaks = np.concatenate([stream.half_breaks for stream in data])
    half_breaks = half_breaks[np.isfinite(half_breaks)]
    print(
        "Erätauon keskimääräinen kesto",
        round(half_breaks.astype(int).mean(), 1),
        "sekuntia",
    )
    game_breaks = np.concatenate([stream.game_breaks for stream in data])
    game_breaks = game_breaks[np.isfinite(game_breaks)]
    print(
        "Pelitauon keskimääräinen kesto",
        _format_to_minutes(game_breaks.astype(int).mean()),
    )


def _format_to_minutes(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds -= minutes * 60

    return f"{minutes} min {round(seconds)} s"
