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

    @property
    def players(self) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the half

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of player names
        """

        if ANONYMIZE:
            names = np.array([throw.player_id for throw in self.throws], dtype=str)
        else:
            names = np.array([throw.player for throw in self.throws], dtype=str)

        return names

    @property
    def throw_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times between the throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        times = np.array([throw.time for throw in self.throws], dtype=np.datetime64)

        return np.diff(times)

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

    @property
    def players(self) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws of the game

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of player names
        """

        return np.concatenate([half.players for half in self.halfs])

    @property
    def throw_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times between the throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([half.throw_times for half in self.halfs])

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

    @property
    def players(self) -> npt.NDArray[np.str_]:
        """
        Names of the players that threw the throws in the stream

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of player names
        """

        return np.concatenate([game.players for game in self.games])

    @property
    def throw_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times between the throws

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            Times
        """

        return np.concatenate([game.throw_times for game in self.games])

    @property
    def kona_times(self) -> npt.NDArray[np.timedelta64]:
        """
        Times it took to pile the konas

        Returns
        -------
        numpy.ndarray of numpy.timedelta64
            1D array of times
        """

        return np.concatenate([game.kona_times for game in self.games])
