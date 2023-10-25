"""Containers for play time data"""
from dataclasses import dataclass, field

import numpy as np
from numpy import typing as npt

ANONYMIZE = False


@dataclass
class Throwtime:
    """
    Container for the data from a single throw

    Attributes
    ----------
    player_id : int
        ID of the player
    player : str
        Name of the player
    time : numpy.datetime64
        Time of the throw
    """

    player_id: int
    player: str
    time: np.datetime64


@dataclass
class Konatime:
    """
    Container for the data from a single kona

    Attributes
    ----------
    time : numpy.datetime64
        Time of the kona
    """

    time: np.datetime64


@dataclass
class Half:
    """
    Container for the data from a half of a game

    Attributes
    ----------
    throws : list of Throwtime
        Throws in the half
    konas : tuple of (Konatime, Konatime)
        Piled konas from the half
    """

    throws: list[Throwtime] = field(default_factory=list)
    konas: tuple[Konatime, Konatime] = field(default_factory=tuple)

    def players(self, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the half

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of str
            1D array of player names
        """

        if ANONYMIZE:
            names = np.array([throw.player_id for throw in self.throws], dtype=str)
        else:
            names = np.array([throw.player for throw in self.throws], dtype=str)

        if difference:
            names = names[1:]

        return names

    def positions(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws of the half

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

    def throw_numbers(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw

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

        if difference:
            numbers = numbers[1:]

        return numbers

    def throw_times(self, difference: bool = False) -> npt.NDArray[np.timedelta64]:
        """
        Timestamps of the throws in the half

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

    @property
    def kona_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        times = []
        for kona_time in self.konas:
            times.append(kona_time.time - self.throws[-1].time)

        return np.array(times, dtype=np.timedelta64)


@dataclass
class Game:
    """
    Container for the data from a single game

    Attributes
    ----------
    halfs : tuple of (Half, Half)
        Half of the game
    """

    halfs: tuple[Half, Half] = field(default_factory=tuple)

    def players(self, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the game

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of str
            1D array of player names
        """

        return np.concatenate([half.players(difference) for half in self.halfs])

    def positions(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws of the game

        Parameters
        ----------
        difference : bool, default False
            Whether to return the positions for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of player positions
        """

        return np.concatenate([half.positions(difference) for half in self.halfs])

    def throw_numbers(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw

        Parameters
        ----------
        difference : bool, default False
            Whether to return the numbers for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of throw numbers
        """

        return np.concatenate([half.throw_numbers(difference) for half in self.halfs])

    def throw_times(self, difference: bool = False) -> npt.NDArray[np.timedelta64]:
        """
        Timestamps of the throws in the game

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([half.throw_times(difference) for half in self.halfs])

    @property
    def kona_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([half.kona_times for half in self.halfs])


@dataclass
class Stream:
    """
    Container for the data from the games played on a single pitch from a single stream

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

    def players(self, difference: bool = False) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws in the stream

        Parameters
        ----------
        difference : bool, default False
            Whether to return the names for the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of player names
        """

        return np.concatenate([game.players(difference) for game in self.games])

    def positions(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Positions of the players that threw the throws in the stream

        Parameters
        ----------
        difference : bool, default False
            Whether to return the positions for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of player positions
        """

        return np.concatenate([game.positions(difference) for game in self.games])

    def throw_numbers(self, difference: bool = False) -> npt.NDArray[np.int_]:
        """
        Whether the throw is the first or second throw

        Parameters
        ----------
        difference : bool, default False
            Whether to return the numbers for the times between throws

        Returns
        -------
        numpy.ndarray of int
            1D array of throw numbers
        """

        return np.concatenate([game.throw_numbers(difference) for game in self.games])

    def throw_times(self, difference: bool = False) -> npt.NDArray[np.timedelta64]:
        """
        Timestamps of the throws in the stream

        Parameters
        ----------
        difference : bool, default False
            Whether to return the times between throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            Times
        """

        return np.concatenate([game.throw_times(difference) for game in self.games])

    @property
    def kona_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times it took to pile the konas in the streams

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([game.kona_times for game in self.games])

    @property
    def game_durations(self) -> npt.NDArray[np.timedelta64]:
        """
        Durations of the games in the stream

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        durations = []
        for game in self.games:
            if np.isfinite(game.kona_times) < 4:
                durations.append(game.throw_times()[-1] - game.throw_times()[0])
            else:
                durations.append(game.kona_times[-1] - game.throw_times()[0])

        return np.array(durations)

    @property
    def game_breaks(self) -> npt.NDArray[np.timedelta64]:
        """
        The break times between the games in the stream

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        break_times = []
        for game_index, game in enumerate(self.games[:-1]):
            break_times.append(
                self.games[game_index + 1].throw_times()[0] - game.kona_times[-1]
            )

        return np.array(break_times)

    @property
    def half_breaks(self) -> npt.NDArray[np.timedelta64]:
        """
        The break times between the halfs in the stream

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        break_times = []
        for game in self.games:
            break_times.append(
                game.halfs[1].throw_times()[0] - game.halfs[0].kona_times[-1]
            )

        return np.array(break_times)
