"""
Modeling and testing models.

This subpackage constructs and tests models for kyykkä play times. The behaviour and
performance of the models can be checked using prior and posterior predictive checks,
and simulation based calibration. The results of checks and inference are automatically
visualized and intermediate results cached for persistence.
"""

import logging

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
