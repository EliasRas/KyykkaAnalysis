"""Containers for play time data"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Throwtime:
    player: str
    time: np.datetime64 | None


@dataclass
class Konatime:
    time: np.datetime64


@dataclass
class Half:
    throws: list[Throwtime] = field(default_factory=list)
    konas: tuple[Konatime | None, Konatime | None] = field(default_factory=tuple)


@dataclass
class Game:
    halfs: tuple[Half, Half] = field(default_factory=tuple)


@dataclass
class Stream:
    url: str
    pitch: str
    games: list[Throwtime] = field(default_factory=list)
