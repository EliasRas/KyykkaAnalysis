"""
Containers for play time data.

This module provides containers for kyykkä play time data with functionalities for
processing raw data.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt


@dataclass
class Throwtime:
    """
    Container for the data from a single throw.

    Throwtime contains the timestamp of when a throw was thrown and information about
    the thrower and the game in which the throw was thrown.

    Attributes
    ----------
    player_id : int
        ID of the player
    player : str
        Name of the player
    time : numpy.datetime64
        Time of the throw
    team : str
        Name of the player's team
    playoffs : bool
        Whether the half is from a playoff game
    """

    player_id: int
    player: str
    time: np.datetime64
    team: str
    playoffs: bool


@dataclass
class Konatime:
    """
    Container for the data from a single kona.

    Konatime contains the timestamp for the time when a kona was completed.

    Attributes
    ----------
    time : numpy.datetime64
        Time of the kona
    """

    time: np.datetime64


@dataclass
class Half:
    """
    Container for the data from a half of a game.

    Half contains the information about the throws thrown in a single half of a kyykkä
    game. Half also provides methods for calculating the times between the events of the
    half (throws and completing konas) and collecting additional information about the
    events and the half.

    Attributes
    ----------
    throws : list of Throwtime
        Throws in the half
    konas : tuple of (Konatime, Konatime)
        Completed konas from the half
    """

    throws: list[Throwtime] = field(default_factory=list)
    konas: tuple[Konatime, Konatime] = field(default_factory=tuple)

    def players(
        self, *, difference: bool = False, anonymize: bool = False
    ) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the half.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws
        anonymize : bool, default False
            Whether to return the player IDs instead of names

        Returns
        -------
        numpy.ndarray of str
            1D array of player names
        """

        if anonymize:
            names = np.array([throw.player_id for throw in self.throws], dtype=str)
        else:
            names = np.array([throw.player for throw in self.throws], dtype=str)

        if difference:
            names = names[1:]

        return names

    def teams(self, *, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the teams that threw the throws of the half.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of str
            1D array of team names
        """

        names = np.array([throw.team for throw in self.throws], dtype=str)
        if difference:
            names = names[1:]

        return names

    def playoffs(self, *, difference: bool = False) -> npt.NDArray[np.bool_]:
        """
        Whether the throws are from playoffs game or not.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the values for the times between throws

        Returns
        -------
        numpy.ndarray of bool
            1D array of values
        """

        names = np.array([throw.playoffs for throw in self.throws], dtype=bool)
        if difference:
            names = names[1:]

        return names

    def positions(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws of the half.

        Collects the positions of the players that threw the throws of the half.
        Position is 1 for the first player and 4 for the last player.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the positions for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of player positions
        """

        throw_numbers = np.remainder(np.arange(len(self.throws)), 8) + 1

        if difference:
            throw_numbers = throw_numbers[1:]

        return np.ceil(throw_numbers / 2)

    def throw_numbers(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw.

        Collects the throw numbers of the throws of the half. First throw of each turn
        of each player is numbered 1 and the second 2.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the numbers for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of throw numbers
        """

        numbers = np.remainder(np.arange(len(self.throws)), 2) + 1
        turn_indices = np.zeros(  # First 2 throws of each player
            min(16, len(self.throws)), int
        )
        if len(self.throws) > 16:  # noqa: PLR2004
            turn_indices = np.concatenate(  # Last 2 throws of each player
                (
                    turn_indices,
                    np.ones(len(self.throws) - 16, int),
                )
            )
        numbers += 2 * turn_indices

        if difference:
            numbers = numbers[1:]

        return numbers

    def throw_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the throws in the half.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        times = np.array([throw.time for throw in self.throws], dtype=np.datetime64)

        if difference:
            times = np.diff(times)

        return times

    def kona_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the completed konas in the half.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        if difference:
            times = [kona_time.time - self.throws[-1].time for kona_time in self.konas]
            times = np.array(times, dtype=np.timedelta64)
        else:
            times = np.array(
                [kona_time.time for kona_time in self.konas], dtype=np.datetime64
            )

        return times

    @property
    def end(self) -> np.datetime64:
        """
        Timestamp for the end of the half.

        Returns
        -------
        numpy.datetime64
            End of the half
        """

        if np.isfinite(self.kona_times()).sum() < 2 and len(self.throws) > 0:  # noqa: PLR2004
            end = self.throws[-1].time
        elif np.isfinite(self.kona_times()).sum() < 2:  # noqa: PLR2004
            end = np.datetime64("NaT")
        else:
            end = self.konas[-1].time

        return end

    @property
    def duration(self) -> np.timedelta64:
        """
        Duration of the half.

        Returns
        -------
        numpy.timedelta64
            Duration of the half
        """

        if len(self.throws) < 32:  # noqa: PLR2004
            duration = np.timedelta64("NaT")
        elif np.isfinite(self.kona_times()).sum() < 2:  # noqa: PLR2004
            duration = self.throw_times()[-1] - self.throw_times()[0]
        else:
            duration = self.kona_times()[-1] - self.throw_times()[0]

        return duration


@dataclass
class Game:
    """
    Container for the data from a single game.

    Attributes
    ----------
    halfs : tuple of (Half, Half)
        Half of the game
    """

    halfs: tuple[Half, Half] = field(default_factory=tuple)

    def players(
        self, *, difference: bool = False, anonymize: bool = False
    ) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the game.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws
        anonymize : bool, default False
            Whether to return the player IDs instead of names

        Returns
        -------
        numpy.ndarray of str
            1D array of player names
        """

        return np.concatenate(
            [
                half.players(difference=difference, anonymize=anonymize)
                for half in self.halfs
            ]
        )

    def teams(self, *, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the teams that threw the throws of the game.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of str
            1D array of team names
        """

        return np.concatenate(
            [half.teams(difference=difference) for half in self.halfs]
        )

    def playoffs(self, *, difference: bool = False) -> npt.NDArray[np.bool_]:
        """
        Whether the throws are from playoffs game or not.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the values for the times between throws

        Returns
        -------
        numpy.ndarray of bool
            1D array of values
        """

        return np.concatenate(
            [half.playoffs(difference=difference) for half in self.halfs]
        )

    def positions(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws of the game.

        Collects the positions of the players that threw the throws of the game.
        Position is 1 for the first player and 4 for the last player.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the positions for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of player positions
        """

        return np.concatenate(
            [half.positions(difference=difference) for half in self.halfs]
        )

    def throw_numbers(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw.

        Collects the throw numbers of the throws of the game. First throw of each turn
        of each player is numbered 1 and the second 2.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the numbers for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of throw numbers
        """

        return np.concatenate(
            [half.throw_numbers(difference=difference) for half in self.halfs]
        )

    def throw_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the throws in the game.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate(
            [half.throw_times(difference=difference) for half in self.halfs]
        )

    def kona_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the piled konas in the game.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate(
            [half.kona_times(difference=difference) for half in self.halfs]
        )

    @property
    def end(self) -> np.datetime64:
        """
        Timestamp for the end of the game.

        Returns
        -------
        numpy.datetime64
            End of the game
        """

        return self.halfs[-1].end

    @property
    def duration(self) -> np.timedelta64:
        """
        Duration of the game.

        Returns
        -------
        numpy.timedelta64
            Duration of the game
        """

        duration = self.half_durations.sum() + self.half_break

        return duration

    @property
    def half_durations(self) -> npt.NDArray[np.timedelta64]:
        """
        Durations of the halfs in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.array([half.duration for half in self.halfs])

    @property
    def half_break(self) -> np.timedelta64:
        """
        The break times between the halfs in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        if len(self.halfs) > 1:
            break_time = self.halfs[1].throw_times()[0] - self.halfs[0].kona_times()[-1]
        else:
            break_time = np.timedelta64("NaT")

        return break_time


@dataclass
class Stream:
    """
    Container for the data from the games played on a single pitch from a single stream.

    Attributes
    ----------
    url : str
        URL of the stream
    pitch : str
        Description of the pitch
    games : list of Game
        Games played on the pitch in the stream
    """

    url: str
    pitch: str
    games: list[Game] = field(default_factory=list)

    def players(
        self, *, difference: bool = False, anonymize: bool = False
    ) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws in the stream.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws
        anonymize : bool, default False
            Whether to return the player IDs instead of names

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of player names
        """

        return np.concatenate(
            [
                game.players(difference=difference, anonymize=anonymize)
                for game in self.games
            ]
        )

    def teams(self, *, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the teams that threw the throws in the stream.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of team names
        """

        return np.concatenate(
            [game.teams(difference=difference) for game in self.games]
        )

    def playoffs(self, *, difference: bool = False) -> npt.NDArray[np.bool_]:
        """
        Whether the throws are from playoffs game or not.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the values for the times between throws

        Returns
        -------
        numpy.ndarray of bool
            1D array of values
        """

        return np.concatenate(
            [game.playoffs(difference=difference) for game in self.games]
        )

    def positions(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws in the stream.

        Collects the positions of the players that threw the throws of the half.
        Position is 1 for the first player and 4 for the last player.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the positions for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of player positions
        """

        return np.concatenate(
            [game.positions(difference=difference) for game in self.games]
        )

    def throw_numbers(self, *, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw.

        Collects the throw numbers of the throws of the half. First throw of each turn
        of each player is numbered 1 and the second 2.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the numbers for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of throw numbers
        """

        return np.concatenate(
            [game.throw_numbers(difference=difference) for game in self.games]
        )

    def throw_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the throws in the stream.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            Times
        """

        return np.concatenate(
            [game.throw_times(difference=difference) for game in self.games]
        )

    def kona_times(
        self, *, difference: bool = False
    ) -> npt.NDArray[np.timedelta64 | np.datetime64]:
        """
        Timestamps of the piled konas in the stream.

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate(
            [game.kona_times(difference=difference) for game in self.games]
        )

    @property
    def end(self) -> np.datetime64:
        """
        Timestamp for the end of the stream.

        Returns
        -------
        numpy.datetime64
            End of the stream
        """

        return self.games[-1].end

    @property
    def game_durations(self) -> npt.NDArray[np.timedelta64]:
        """
        Durations of the games in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.array([game.duration for game in self.games])

    @property
    def half_durations(self) -> npt.NDArray[np.timedelta64]:
        """
        Durations of the halfs in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([game.half_durations for game in self.games])

    @property
    def game_breaks(self) -> npt.NDArray[np.timedelta64]:
        """
        The break times between the games in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        break_times = []
        for game_index, game in enumerate(self.games[:-1]):
            break_times.append(
                self.games[game_index + 1].throw_times()[0] - game.kona_times()[-1]
            )

        return np.array(break_times, np.datetime64)

    @property
    def half_breaks(self) -> npt.NDArray[np.timedelta64]:
        """
        The break times between the halfs in the stream.

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.array([game.half_break for game in self.games])


class ModelData:
    """
    Container for the data needed by the throw time models.

    ModelData collects the processed play time data in a form that can be used when
    constructing throw time models.

    Attributes
    ----------
    throw_times : numpy.ndarray of int
        Throw times
    players : numpy.ndarray of str
        Names of the players for each throw
    player_ids : numpy.ndarray of int
        IDs of the players for each throw
    player_names : numpy.ndarray of str
        Names of the players ordered by the corresponding player IDs. "" for the IDs
        that are not present in the data
    first_throw : numpy.ndarray of bool
        Whether the throw was the player's first throw in a turn
    """

    def __init__(self, data: list[Stream]) -> None:
        """
        Container for the data needed by the throw time models.

        Parameters
        ----------
        data : list of Stream
            Throw time data
        """

        throw_times: npt.NDArray[np.float64] = np.concatenate(
            [stream.throw_times(difference=True) for stream in data]
        )
        players = np.concatenate([stream.players(difference=True) for stream in data])
        self.player_ids: npt.NDArray[np.int_] = np.concatenate(
            [stream.players(difference=True, anonymize=True) for stream in data]
        ).astype(int)
        player_names = []
        for player_id in range(self.player_ids.max() + 1):
            if player_id in self.player_ids:
                player_names.append(players[self.player_ids == player_id][0])
            else:
                player_names.append("")
        self.player_names = np.array(player_names, str)

        throw_number = np.concatenate(
            [stream.throw_numbers(difference=True) for stream in data]
        )
        self.first_throw: npt.NDArray[np.bool_] = (throw_number == 0) | (
            throw_number == 3  # noqa: PLR2004
        )

        valid_times = np.isfinite(throw_times)
        self.throw_times = throw_times[valid_times].astype(float)
        self.player_ids = self.player_ids[valid_times]
        self.first_throw = self.first_throw[valid_times]
