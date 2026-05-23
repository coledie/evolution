"""
Run the evolutionary algorithm on the 8-queens problem.

Uses MetaNQueens (fitness = 28 - clashes; solution when clashes == 0).
"""
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


def main():
    np.random.seed(0)

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
    total_evaluations = 0  # every call to get_fitness (including cache hits)
    fresh_evaluations = 0  # only the ones that actually ran the fitness function

    # Wrap the fitness function so we can count fresh evaluations (cache misses).
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
        print(f"epoch {population.epoch:3d} | max fitness {epoch_best:5.1f} | best so far {best_fitness:5.1f}")
        if population.terminated:
            break
        population.update(fitnesses)

    print()
    print(f"Best fitness: {best_fitness} (28 = solved)")
    print(f"Total genotypes tested (incl. cache hits): {total_evaluations}")
    print(f"Unique genotypes evaluated (cache misses):  {fresh_evaluations}")
    print(f"Best genotype: {best_genotype}")

    if best_genotype is not None:
        render_board(best_genotype)


def render_board(genotype, n=8):
    board = [["." for _ in range(n)] for _ in range(n)]
    for letter in "abcdefgh"[:n]:
        x = int(genotype[letter + "x"])
        y = int(genotype[letter + "y"])
        board[y][x] = "Q" if board[y][x] == "." else "X"
    print()
    for row in reversed(board):
        print(" ".join(row))


if __name__ == "__main__":
    main()
