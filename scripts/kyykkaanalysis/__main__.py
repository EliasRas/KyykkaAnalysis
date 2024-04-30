"""Provides an entry point for kyykkä play time analysis."""

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import structlog
from structlog import dev, processors

from kyykkaanalysis.data.data_classes import Stream
from kyykkaanalysis.data.data_description import print_description
from kyykkaanalysis.data.data_reading import read_times
from kyykkaanalysis.figures import data as data_figures
from kyykkaanalysis.model import inference, model_checks
from kyykkaanalysis.model.modeling import ModelType

_LOG = structlog.get_logger(__name__)


def main() -> None:
    """Run analysis."""

    _configure_log()
    _LOG.info("Program started.")

    args = _parse_arguments()
    data = read_times(args.input_file, args.team_file)

    if "data" in args.pipelines:
        _data_pipeline(data, args)
    if "prior" in args.pipelines:
        _prior_pipeline(data, args)
    if "sbc" in args.pipelines:
        _LOG.info("Running simulation based calibration pipeline.")
        model_checks.fake_data_simulation(
            data, args.figure_directory, args.cache_directory
        )
        _LOG.info("Simulation based calibration pipeline finished.")
    if "posterior" in args.pipelines:
        _LOG.info("Running posterior inference pipeline.")
        inference.fit_model(data, args.figure_directory, args.cache_directory)
        _LOG.info("Posterior inference pipeline finished.")


def _configure_log() -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.INFO, force=True
    )

    configured_processors = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        processors.StackInfoRenderer(),
        dev.set_exc_info,
        processors.format_exc_info,
        processors.TimeStamper(fmt="iso", utc=True),
        processors.TimeStamper(utc=True, key="UNIX timestamp"),
        processors.CallsiteParameterAdder(
            {
                processors.CallsiteParameter.FUNC_NAME,
                processors.CallsiteParameter.LINENO,
            }
        ),
    ]
    if sys.stderr.isatty():
        configured_processors.append(dev.ConsoleRenderer())
    else:
        configured_processors.append(processors.JSONRenderer())

    structlog.configure(
        processors=configured_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input_file",
        help="Path to the CSV file which contains the input data",
        type=Path,
    )
    parser.add_argument(
        "--team_file",
        help="Path to the CSV file which contains the teams for the player",
        type=Path,
    )
    parser.add_argument(
        "--cache_directory",
        help="Path to the directory to which the cached data is saved",
        type=Path,
    )
    parser.add_argument(
        "--figure_directory",
        help="Path to the directory to which the visualizations are saved",
        type=Path,
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="List of pipelines to run",
        type=str,
    )
    parser.add_argument(
        "--test",
        nargs="+",
        type=str,
    )

    args = parser.parse_args()
    _LOG.debug(
        "Command line arguments read.",
        input_file=args.input_file,
        team_file=args.team_file,
        cache_directory=args.cache_directory,
        figure_directory=args.figure_directory,
        pipelines=args.pipelines,
    )

    return args


def _data_pipeline(data: list[Stream], args: Namespace) -> None:
    _LOG.info("Running data pipeline.")
    print_description(data)
    data_figures.time_distributions(data, args.figure_directory / "Data")
    data_figures.averages(data, args.figure_directory / "Data")
    _LOG.info("Data pipeline finished.")


def _prior_pipeline(data: list[Stream], args: Namespace) -> None:
    _LOG.info("Running prior pipeline.")
    model_checks.check_priors(
        data,
        args.figure_directory / "Prior",
        args.cache_directory,
        model_type=ModelType.GAMMA,
    )
    model_checks.check_priors(
        data,
        args.figure_directory / "Prior",
        args.cache_directory,
        model_type=ModelType.NAIVE,
    )
    model_checks.check_priors(
        data,
        args.figure_directory / "Prior",
        args.cache_directory,
        model_type=ModelType.INVGAMMA,
    )
    model_checks.check_priors(
        data,
        args.figure_directory / "Prior",
        args.cache_directory,
        model_type=ModelType.NAIVEINVGAMMA,
    )
    _LOG.info("Prior pipeline finished.")


if __name__ == "__main__":
    main()
