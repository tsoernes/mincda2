#! /usr/bin/env python3
import argparse
import logging
from strats import FAStrat, RS_SARSA
from grid import Grid, FixedGrid
from eventgen import EventGen


def get_args(defaults=False) -> dict:
    """
    If defaults, then return default arguments instead of parsing from cmd line
    """
    parser = argparse.ArgumentParser()

    strats = {"fca": FAStrat, "rs_sarsa": RS_SARSA}
    parser.add_argument("strat", type=str, choices=strats.keys(), default="rs_sarsa")

    parser.add_argument("--call_duration", type=int, help="in minutes", default=3)
    parser.add_argument(
        "--hoff_call_duration",
        type=int,
        help="handoff call duration, in minutes",
        default=1,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--call_rate", type=float, help="in calls per minute", default=200 / 60
    )
    group.add_argument(
        "--call_rateh", type=float, help="in calls per hour"
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
        "--alpha", type=float, help="Learning rate for RL", default=0.05
    )
    parser.add_argument(
        '--alpha_decay',
        type=float,
        help="(RL/Table) factor by which alpha is multiplied each iteration",
        default=0.999_999_9)
    # parser.add_argument(
    #     "--alpha_avg", type=float, help="Learning rate for average reward", default=0.06
    # )
    # parser.add_argument(
    #     "--alpha_grad",
    #     type=float,
    #     help="Learning rate for TDC gradient corrections",
    #     default=5e-6,
    # )
    parser.add_argument('--gamma', type=float, help="(RL) discount factor", default=0.975)
    parser.add_argument(
        '--epsilon',
        '-eps',
        dest='epsilon',
        type=float,
        help="(RL) exploration hyperparameter",
        default=5)
    parser.add_argument(
        '-edec',
        '--epsilon_decay',
        type=float,
        help="(RL) factor by which epsilon is multiplied each iteration",
        default=0.999_995)
    parser.add_argument(
        '--eps_log_decay',
        type=int,
        help="(RL) Decay epsilon a la Lilith instead of exponentially (give s parameter)",
        default=256)

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
    parser.add_argument(
        "--no_sanity_check",
        action="store_true",
        default=False,
    )
    # Lilith preset
    # if pp['lilith'] or pp['lilith_noexp']:
    #     pp['alpha'] = 0.05
    #     pp['alpha_decay'] = 1
    #     pp['target'] = 'discount'
    #     pp['gamma'] = 0.975
    # if pp['lilith']:
    #     pp['eps_log_decay'] = 256
    #     pp['epsilon'] = 5

    if defaults:
        args = vars(parser.parse_args(["rs_sarsa"]))
    else:
        args = vars(parser.parse_args())

    # Convert e.g. {"no_sanity_check": True} to {"sanity_check": False}
    for arg, val in args.copy().items():
        if arg.startswith("no_"):
            args[arg[3:]] = not val
            del args[arg]

    args["strat"] = strats[args["strat"]]

    if args["call_rateh"]:
        args["call_rate"] = args["call_rateh"] / 60
    return args


def main(defaults=False):
    # Get problem parameters
    pp = get_args(defaults=defaults)

    # Initialize a logger
    logging.basicConfig(level=pp["log_level"], format="%(message)s")
    logger = logging.getLogger("")
    logger.info(f"Starting simulation with params {pp}")

    # Initializse a event generator
    eventgen = EventGen(**pp)

    # Initialize an agent (i.e. a strategy) for allocation channels
    strat = pp["strat"](
        pp, eventgen=eventgen, sanity_check=True, logger=logger
    )

    # Start simulation
    strat.simulate()


if __name__ == "__main__":
    main(defaults=False)
