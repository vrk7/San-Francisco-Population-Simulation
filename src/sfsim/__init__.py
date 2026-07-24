"""sfsim — a mini generative-agents simulation of San Francisco residents."""

import argparse

from sfsim.pipeline import run


def main() -> None:
    """CLI entry point: run the SF population simulation end to end."""
    parser = argparse.ArgumentParser(
        prog="sfsim",
        description="Simulate 30 SF residents voting on Prop X and benchmark vs. reality.",
    )
    parser.add_argument(
        "--no-extras", action="store_true",
        help="skip the social-influence and reflection stages (run Parts 1-3 only)",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="skip the memory-off baseline comparison",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="cap the number of agents (for a quick demo); 0 = all 30",
    )
    parser.add_argument(
        "--pause", type=float, default=0.0, metavar="SECONDS",
        help="delay between LLM calls (rate-limit courtesy for hosted providers)",
    )
    args = parser.parse_args()
    run(
        extras=not args.no_extras,
        compare=not args.no_compare,
        limit=args.limit,
        pause=args.pause,
    )
