"""Analysis of kyykka play times"""
from pathlib import Path
from argparse import ArgumentParser, Namespace


def main():
    """
    Run analysis
    """

    args = _parse_arguments()


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Path to the CSV file which contains the input data",
        type=Path,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
