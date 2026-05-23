"""
Base meta reinforcement learning environment template.

MetaNQueens - place N queens on a chessboard without conflicts.
EvolveNetwork - tune SNN hyperparameters via evolutionary search.
"""
from ..module import Module, Key
import numpy as np
from .backends.single import SingleProcessBackend
from .series import Series


class MetaRL(Module):
    """
    Base meta reinforcement learning environment template.

    Parameters
    ----------
    preset: str, default=None
        Configuration preset key.
    kwargs: dict
        Game parameters for NECESSARY_KEYS.
    """

    NECESSARY_KEYS = []
    GENOTYPE_CONSTRAINTS = {}
    PRESETS = {}

    def __init__(self, preset=None, **kwargs):
        self._params = {}
        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)
        self._params.update(
            {
                key.name if hasattr(key, "name") else key: kwargs[key]
                for key in self.NECESSARY_KEYS
                if key in kwargs
            }
        )
        super().__init__(**self._params)
        self._add_values(self._params, dest=self._params, prefix="")

    @property
    def params(self):
        return self._params

    def reset(self):
        pass

    def get_fitness(self, genotype):
        raise NotImplementedError(f"get_fitness not implemented for {type(self)}!")

    def step(self, action, **kwargs):
        fitness, done = self.get_fitness(action, **kwargs)
        return None, fitness, done, {}

    def close(self):
        pass

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.get_state()


class MetaNQueens(MetaRL):
    """
    Place N queens on a chessboard without any attacking each other.

    92 distinct solutions out of 4 billion possibilities with 8 queens.
    Genotypes: for each queen i, keys {letter}x and {letter}y in [0, 7].
    """

    NECESSARY_KEYS = MetaRL.extend_keys(
        [Key("n_queens", "{1..8} Number of queens to place.", int, default=8)]
    )
    GENOTYPE_CONSTRAINTS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._n_queens > 8 or self._n_queens < 1:
            raise ValueError(f"n_queens must be in [1, 8], not {self._n_queens}!")
        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h"][: self._n_queens]
        keys = [first + second for second in ["x", "y"] for first in self.letters]
        self.GENOTYPE_CONSTRAINTS = {key: list(range(8)) for key in keys}

    @staticmethod
    def setup_game():
        return np.zeros(8), np.zeros(8), np.zeros(15), np.zeros(15)

    @staticmethod
    def run_move(board, move):
        horizontals, verticals, ldiagonals, rdiagonals = board
        x, y = move
        horizontals[x] += 1
        verticals[y] += 1
        ldiagonals[x + y] += 1
        rdiagonals[7 - x + y] += 1
        return horizontals, verticals, ldiagonals, rdiagonals

    def get_fitness(self, genotype):
        board = self.setup_game()
        for letter in self.letters:
            board = self.run_move(board, (genotype[letter + "x"], genotype[letter + "y"]))
        clashes = sum(np.sum(item[item > 1] - 1) for item in board)
        fitness = 28 - clashes
        return fitness, clashes == 0


class EvolveNetwork(MetaRL):
    """
    Tune spiking neural network parameters on an RL game via evolutionary search.

    GENOTYPE_CONSTRAINTS are set by the genotype_constraints init parameter.
    """

    NECESSARY_KEYS = MetaRL.extend_keys(
        [
            Key("training_loop", "Pre-configured training loop to evaluate fitness."),
            Key("genotype_constraints", "Constraints for each parameter to evolve.", dict),
            Key("static_updates", "Static parameter updates for the training loop.", default=None),
            Key("n_reruns", "Number of times to rerun each experiment.", int, default=2),
            Key("win_fitness", "Fitness threshold to terminate MetaRL.", float),
            Key("fitness_getter", "f(net, game, results, info)->float"),
            Key("fitness_aggregator", "f([fitness, ..])->float", default=np.mean),
        ]
    )
    GENOTYPE_CONSTRAINTS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GENOTYPE_CONSTRAINTS = self._genotype_constraints

    def get_fitness(self, genotype):
        training_loop = self._training_loop.copy()
        training_loop.reset(**genotype, **self.params)
        series = Series(training_loop, self._static_updates, backend=SingleProcessBackend())

        tracking = []
        for experiment in series:
            for _ in range(self._n_reruns):
                network, game, results, info = experiment()
                tracking.append(self._fitness_getter(network, game, results, info))

        fitness = self._fitness_aggregator(tracking)
        return fitness, fitness >= self._win_fitness
