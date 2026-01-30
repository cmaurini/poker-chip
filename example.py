#!/usr/bin/env python3
"""
Example script showing how to use the poker_chip_lib library

This demonstrates the basic workflow for running phase-field fracture simulations
"""

import sys
import argparse
from pathlib import Path

# If installed as a package, simply:
# from poker_chip_lib import poker_chip
# 
# Otherwise for development:
# sys.path.insert(0, str(Path(__file__).parent))
# from poker_chip_lib import poker_chip

def main():
    """Run poker chip simulation example"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Poker chip phase-field fracture simulation example"
    )
    parser.add_argument(
        "--gdim",
        type=int,
        default=2,
        help="Geometric dimension (2 or 3)"
    )
    parser.add_argument(
        "--ell",
        type=float,
        default=0.01,
        help="Internal length scale"
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=20,
        help="Number of load steps"
    )
    parser.add_argument(
        "--loadmax",
        type=float,
        default=1.0,
        help="Maximum load"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="example_simulation",
        help="Output directory name"
    )
    
    args = parser.parse_args()
    
    print(f"Starting poker chip simulation:")
    print(f"  Dimension: {args.gdim}D")
    print(f"  Internal length scale: {args.ell}")
    print(f"  Load steps: {args.nstep}")
    print(f"  Maximum load: {args.loadmax}")
    print(f"  Output: {args.output_name}")
    print()
    
    # Import the main simulation module
    from poker_chip_lib import poker_chip
    
    # You can now use the poker_chip module directly
    # The main entry point is poker_chip.main() or similar
    # (Details depend on the actual structure of poker_chip.py)
    
    print("To run the full simulation, execute:")
    print(f"  python -m poker_chip_lib.poker_chip --gdim {args.gdim} --ell {args.ell}")

if __name__ == "__main__":
    main()
