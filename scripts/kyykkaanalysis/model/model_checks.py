"""Checking priors and model definition"""
from pathlib import Path

from .modeling import ThrowTimeModel
from ..data.data_classes import Stream, ModelData
from ..figures.prior import parameter_distributions


def check_priors(data: list[Stream], figure_directory: Path):
    """
    Sample data from prior distribution and analyze it

    Parameters
    ----------
    data : list of Stream
        Throw time data
    figure_directory : Path
        Path to the directory in which the figures are saved
    """

    model = ThrowTimeModel(ModelData(data))
    samples = model.sample_prior(10000)
    parameter_distributions(
        samples, model.data.player_ids, model.data.first_throw, figure_directory
    )