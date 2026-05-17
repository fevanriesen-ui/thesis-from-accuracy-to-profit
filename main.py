"""
main.py
===============================================================================
Full Pipeline Runner — MSc Thesis: From Accuracy to Profit
F.E. van Riesen | Tilburg University | Data Science & Society | 2026
===============================================================================

Runs the complete pipeline or selected steps.

USAGE
-----
  python main.py                 # run all 4 steps
  python main.py 3 4             # run only steps 3 and 4
  python main.py --from 2        # run steps 2, 3, 4
===============================================================================
"""

import argparse
import os
import sys
import time
from importlib import import_module

# The pipeline scripts live in src/; add it to sys.path so they can be
# imported by name. (Filenames starting with a digit cannot be referenced
# via standard `import` syntax, so importlib + sys.path is the cleanest
# route — see src/01_build_dataset.py etc.)
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

STEPS = [
    (1, "01_build_dataset",     "Building dataset"),
    (2, "02_train_models",      "Training models"),
    (3, "03_evaluation",        "Evaluating & generating figures"),
    (4, "04_significance_test", "Running significance tests"),
]


def main():
    parser = argparse.ArgumentParser(
        description="MSc Thesis Pipeline — From Accuracy to Profit")
    parser.add_argument("steps", nargs="*", type=int,
                        help="Step number(s) to run (1-4). Default: all.")
    parser.add_argument("--from", dest="from_step", type=int, default=None,
                        help="Run from this step onwards (e.g. --from 2).")
    args = parser.parse_args()

    if args.from_step is not None:
        selected = [s for s in STEPS if s[0] >= args.from_step]
    elif args.steps:
        selected = [s for s in STEPS if s[0] in args.steps]
    else:
        selected = STEPS

    if not selected:
        parser.error("No valid steps selected. Choose from 1-4.")

    print("=" * 70)
    print("MSc Thesis Pipeline — From Accuracy to Profit")
    print("F.E. van Riesen | Tilburg University | 2026")
    print("=" * 70)

    t_total = time.time()

    for step_num, module_name, description in selected:
        print(f"\n{'=' * 70}")
        print(f"  Step {step_num}/4 — {description}")
        print(f"{'=' * 70}\n")

        t_step = time.time()
        module = import_module(module_name)
        module.main()
        elapsed = time.time() - t_step
        print(f"\n  [{module_name}] finished in {elapsed:.1f}s")

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete — total time: {elapsed_total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
