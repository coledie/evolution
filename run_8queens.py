"""
Run the evolutionary algorithm on the 8-queens problem.

Uses MetaNQueens (fitness = 28 - clashes; solution when clashes == 0).

Usage:
    python run_8queens.py [seed ...]

If no seeds are given, seed 0 is used. When multiple seeds are given,
each is run independently and the final boards are printed side-by-side.
"""
import argparse
import os
import sys

# Make the `evolution` package importable by adding its parent dir to sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import numpy as np

from evolution.meta import MetaNQueens, Population
from evolution.meta.backends.single import SingleProcessBackend


def run(seed, verbose=True):
    np.random.seed(seed)

    game = MetaNQueens(n_queens=8)

    population = Population(
        game,
        backend=SingleProcessBackend(),
        n_storing=500,
        n_agents=200,
        n_epoch=100,
        mutate_eligable_pct=0.5,
        max_age=5,
        random_rate=0.1,
        survivor_rate=0.2,
        mutation_rate=0.5,
        crossover_rate=0.2,
        logging=False,
    )

    best_fitness = -np.inf
    best_genotype = None
    total_evaluations = 0
    fresh_evaluations = 0
    solved_epoch = None

    original_get_fitness = population.get_fitness

    def counting_get_fitness(genotype):
        nonlocal fresh_evaluations
        fresh_evaluations += 1
        return original_get_fitness(genotype)

    population.get_fitness = counting_get_fitness

    while not population.terminated:
        total_evaluations += len(population.population)
        fitnesses = population.evaluate()
        epoch_best = max(fitnesses)
        if epoch_best > best_fitness:
            best_fitness = epoch_best
            best_genotype = population.population[int(np.argmax(fitnesses))]
            if best_fitness == 28 and solved_epoch is None:
                solved_epoch = population.epoch
        if verbose:
            print(f"[seed={seed}] epoch {population.epoch:3d} | max {epoch_best:5.1f} | best {best_fitness:5.1f}")
        if population.terminated:
            break
        population.update(fitnesses)

    return {
        "seed": seed,
        "best_fitness": best_fitness,
        "best_genotype": best_genotype,
        "total_evaluations": total_evaluations,
        "fresh_evaluations": fresh_evaluations,
        "solved_epoch": solved_epoch,
    }


def board_rows(genotype, n=8):
    """Return n strings (top row first) representing the board."""
    board = [["." for _ in range(n)] for _ in range(n)]
    for letter in "abcdefgh"[:n]:
        x = int(genotype[letter + "x"])
        y = int(genotype[letter + "y"])
        board[y][x] = "Q" if board[y][x] == "." else "X"
    return [" ".join(row) for row in reversed(board)]


def render_boards_side_by_side(results, gap="    "):
    cols = []
    for r in results:
        if r["best_genotype"] is None:
            continue
        rows = board_rows(r["best_genotype"])
        header = f"seed={r['seed']}"
        width = max(len(header), max(len(row) for row in rows))
        col = [header.center(width)] + [row.ljust(width) for row in rows]
        cols.append(col)
    if not cols:
        return ""
    return "\n".join(gap.join(parts) for parts in zip(*cols))


def main():
    parser = argparse.ArgumentParser(description="Evolve solutions to 8-queens.")
    parser.add_argument(
        "seeds",
        nargs="*",
        type=int,
        default=[0],
        help="One or more integer seeds (default: 0).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-epoch logging.",
    )
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        result = run(seed, verbose=not args.quiet)
        results.append(result)
        print()
        print(f"[seed={seed}] best fitness {result['best_fitness']} "
              f"(solved at epoch {result['solved_epoch']}) | "
              f"total tested {result['total_evaluations']} | "
              f"unique {result['fresh_evaluations']}")
        print()

    print("Final boards:")
    print()
    print(render_boards_side_by_side(results))


if __name__ == "__main__":
    main()
