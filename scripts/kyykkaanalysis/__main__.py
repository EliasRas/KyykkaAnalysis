"""Analysis of kyykka play times"""
from pathlib import Path
from argparse import ArgumentParser, Namespace

from kyykkaanalysis.data.data_reading import read_times


def main():
    """
    Run analysis
    """

    args = _parse_arguments()
    data = read_times(args.input_file)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Path to the CSV file which contains the input data",
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
