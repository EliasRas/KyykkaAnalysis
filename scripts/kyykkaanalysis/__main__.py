"""Analysis of kyykka play times"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from kyykkaanalysis.data.data_description import print_description
from kyykkaanalysis.data.data_reading import read_times
from kyykkaanalysis.figures import data as data_figures
from kyykkaanalysis.model import inference, model_checks
from kyykkaanalysis.model.modeling import ModelType


def main():
    """
    Run analysis
    """

    args = _parse_arguments()
    data = read_times(args.input_file, args.team_file)

    print_description(data)

    data_figures.time_distributions(data, args.figure_directory / "Data")
    data_figures.averages(data, args.figure_directory / "Data")

    model_checks.check_priors(
        data, args.figure_directory / "Prior", args.cache_directory, ModelType.GAMMA
    )
    model_checks.check_priors(
        data, args.figure_directory / "Prior", args.cache_directory, ModelType.NAIVE
    )
    model_checks.check_priors(
        data, args.figure_directory / "Prior", args.cache_directory, ModelType.INVGAMMA
    )
    model_checks.check_priors(
        data,
        args.figure_directory / "Prior",
        args.cache_directory,
        ModelType.NAIVEINVGAMMA,
    )
    model_checks.fake_data_simulation(data, args.figure_directory, args.cache_directory)


    inference.fit_model(data, args.figure_directory, args.cache_directory)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Path to the CSV file which contains the input data",
        type=Path,
    )
    parser.add_argument(
        "team_file",
        help="Path to the CSV file which contains the teams for the player",
        type=Path,
    )
    parser.add_argument(
        "cache_directory",
        help="Path to the directory to which the cached data is saved",
        type=Path,
    )
    parser.add_argument(
        "figure_directory",
        help="Path to the directory to which the visualizations are saved",
        type=Path,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
