#! /usr/bin/env python3
import argparse
import logging
from strats import FAStrat
from grid import Grid, FixedGrid
from eventgen import EventGen


def get_args(defaults=False) -> dict:
    """
    If defaults, then return default arguments instead of parsing from cmd line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--call_duration", type=int, help="in minutes", default=3)
    parser.add_argument(
        "--hoff_call_duration",
        type=int,
        help="handoff call duration, in minutes",
        default=1,
    )
    parser.add_argument(
        "--call_rate", type=float, help="in calls per minute", default=200 / 60
    )
    parser.add_argument(
        "-phoff",
        "--p_handoff",
        type=float,
        nargs="?",
        help="handoff probability.",
        default=None,
        const=0.15,
    )
    parser.add_argument(
        "--n_events",
        dest="n_events",
        type=int,
        help="number of events to simulate",
        default=10_000,
    )
    parser.add_argument(
        "--log_iter",
        dest="log_iter",
        type=int,
        help="Show blocking probability every 'log__iter' iterations",
        default=1_000,
    )
    parser.add_argument(
        "--alpha", type=float, help="Learning rate for neural network", default=2.52e-6
    )
    parser.add_argument(
        "--alpha_avg", type=float, help="Learning rate for average reward", default=0.06
    )
    parser.add_argument(
        "--alpha_grad",
        type=float,
        help="Learning rate for TDC gradient corrections",
        default=5e-6,
    )

    parser.add_argument("--rows", type=int, help="number of rows in grid", default=7)
    parser.add_argument("--cols", type=int, help="number of columns in grid", default=7)
    parser.add_argument("--n_channels", type=int, help="number of channels", default=70)

    parser.add_argument(
        "--log_level",
        type=int,
        choices=[10, 20, 30],
        help="10: Debug,\n20: Info,\n30: Warning",
        default=20,
    )

    if defaults:
        args = vars(parser.parse_args([]))
    else:
        args = vars(parser.parse_args())
    return args


def main(is_main=False):
    # Get problem parameters
    pp = get_args(defaults=is_main)

    # Initialize a logger
    logging.basicConfig(level=pp["log_level"], format="%(message)s")
    logger = logging.getLogger("")
    logger.info(f"Starting simulation with params {pp}")

    # Initialize a caller environment grid
    grid = FixedGrid(logger=logger, **pp)

    # Initializse a event generator
    eventgen = EventGen(**pp)

    # Initialize an agent (i.e. a strategy) for allocation channels
    strat = FAStrat(
        pp, grid=grid, eventgen=eventgen, sanity_check=True, logger=logger
    )

    # Start simulation
    strat.simulate()


if __name__ == "__main__":
    main(is_main=True)
