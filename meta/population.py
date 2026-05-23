"""
An evolving population of genotypes for optimizing hyperparameters.

Examples
--------

.. code-block:: python

    metagame = EvolveNetwork(GenericLoop(network, game, **params), **metagame_config)
    population = Population(metagame, **pop_config)

    while not population.terminated:
        fitness = population.evaluate()
        population.update(fitness)
        print(f"{population.epoch} - Max fitness: {max(fitness)}")
"""
import os
from copy import copy, deepcopy
import numpy as np
from ..module import Module, Key
from .backends.default import MultiprocessBackend
from ..logging import log, MultiLogger


class GenotypeMapping(Module):
    """
    Cache genotype-fitness matchings.
    """

    def __init__(self, n_storing):
        self.n_storing = n_storing
        self.genotypes = []
        self.fitnesses = []

    def __getitem__(self, genotype):
        genotype_no_age = copy(genotype)
        if "_age" in genotype_no_age:
            del genotype_no_age["_age"]
        if genotype_no_age not in self.genotypes:
            return None
        idx = self.genotypes.index(genotype_no_age)
        fitness = self.fitnesses[idx]
        self.update(genotype, fitness)
        return fitness

    def update(self, genotype, fitness):
        if not self.n_storing:
            return
        genotype_no_age = copy(genotype)
        if "_age" in genotype_no_age:
            del genotype_no_age["_age"]
        self.genotypes.append(genotype_no_age)
        self.fitnesses.append(fitness)
        assert len(self.genotypes) == len(self.fitnesses)
        if len(self.genotypes) >= self.n_storing:
            self.genotypes = self.genotypes[-self.n_storing :]
            self.fitnesses = self.fitnesses[-self.n_storing :]


def run(fitness_func, cache, genotype, log_fn, filename):
    fitness = cache[genotype]
    if fitness is not None:
        terminate = False
    else:
        fitness, terminate = fitness_func(genotype)

    if filename:
        log_fn(None, None, results={"fitness": fitness, "filename": filename}, info=genotype, filename=filename)

    cache.update(genotype, fitness)
    return fitness, terminate


def checkpoint_population(population, folder="."):
    from pickle import dump as pickledump

    if folder:
        try:
            os.makedirs(folder)
            print(f"Created directory {folder}!")
        except FileExistsError:
            pass
    file_header = population.multilogger.prefix if hasattr(population, "multilogger") else ""
    filename = f"{file_header}~EPOCH-({population.epoch:03d}).obj"
    with open(os.path.join(folder, filename), "wb") as file:
        pickledump(population, file)


def read_population(folder="."):
    from pickle import load as pickleload

    relevant_filenames = [f for f in os.listdir(folder) if "EPOCH" in f]
    if not relevant_filenames:
        raise ValueError(f"Could not find previous EPOCH data in {folder}!")
    relevant_filenames.sort()
    with open(os.path.join(folder, relevant_filenames[-1]), "rb") as file:
        return pickleload(file)


class Population(Module):
    """
    An evolving population of genotypes.

    Parameters
    ----------
    game: MetaRL
        MetaRL game to evolve agents for.
    backend: MetaBackend, default=MultiprocessBackend(max_process)
        Backend to execute experiments with.
    max_process: int, default=16
        Number of separate processes for default backend.
    kwargs: dict
        Configuration (see NECESSARY_KEYS).
    """

    NECESSARY_KEYS = [
        Key("n_storing", "Number of genotypes to cache.", int),
        Key("n_agents", "Agents per epoch.", (int, list, tuple, np.ndarray)),
        Key("n_epoch", "Number of epochs (unused if n_agents is iterable).", int, default=9999),
        Key("mutate_eligable_pct", "(0, 1] Pct of prev agents eligible to mutate.", float),
        Key("max_age", "Max age before removal from gene pool.", int),
        Key("random_rate", "(0, 1) Fraction of population generated randomly.", float),
        Key("survivor_rate", "(0, 1) Fraction of population preserved per epoch.", float),
        Key("mutation_rate", "(0, 1) Fraction of population mutated per epoch.", float),
        Key("crossover_rate", "(0, 1) Fraction of population crossed over per epoch.", float),
        Key("logging", "Whether to log.", bool, default=True),
        Key("log_fn", "f(n, g, r, i, filename) Logging function.", default=log),
        Key("folder", "Folder to save logs.", str, default="log"),
    ]

    def __init__(self, game, backend=None, max_process=16, **config):
        super().__init__(**config)

        self.genotype_constraints = game.GENOTYPE_CONSTRAINTS
        self.get_fitness = game.get_fitness
        self.backend = backend or MultiprocessBackend(max_process)

        if isinstance(self._n_agents, (list, tuple, np.ndarray)):
            self.n_agents = list(self._n_agents)
        else:
            self.n_agents = [self._n_agents for _ in range(self._n_epoch)]

        self.epoch = 0
        self.terminated = False

        self.cache = GenotypeMapping(self._n_storing)
        self.population = [self._random() for _ in range(self.n_agents[self.epoch])]

        if self._mutate_eligable_pct == 0:
            raise ValueError("mutate_eligable_pct cannot be 0!")

        self._normalize_rates()
        if self._logging:
            self._setup_logging(config, game.params)

    def _normalize_rates(self):
        total = self._random_rate + self._survivor_rate + self._mutation_rate + self._crossover_rate
        if not total:
            raise ValueError("Need nonzero value for survivor, mutation or crossover rate.")
        self._random_rate /= total
        self._survivor_rate /= total
        self._mutation_rate /= total
        self._crossover_rate /= total

    def _setup_logging(self, pop_params, game_params):
        self.multilogger = MultiLogger(folder=self._folder)
        info = {"population_config": pop_params, "metagame_info": game_params}
        self.multilogger.summarize(results=None, info=info)

    def __len__(self):
        return len(self.population)

    def _genotype_dist(self, genotype1, genotype2):
        total = 0
        for key in self.genotype_constraints.keys():
            if isinstance(genotype1[key], (list, tuple)):
                for i in range(len(genotype1[key])):
                    total += (genotype1[key][i] - genotype2[key][i]) ** 2
            else:
                total += (genotype1[key] - genotype2[key]) ** 2
        return total ** 0.5

    def _random(self):
        eval_constraint = (
            lambda cons: np.random.uniform(*cons)
            if isinstance(cons, tuple)
            else cons[np.random.choice(len(cons))]
        )
        genotype = {key: eval_constraint(cons) for key, cons in self.genotype_constraints.items()}
        genotype["_age"] = 0
        return genotype

    def _mutate(self, genotypes):
        if not isinstance(genotypes, (list, np.ndarray)):
            genotypes = [genotypes]
        new_genotypes = []
        for genotype in genotypes:
            new_genotype = deepcopy(genotype)
            key = np.random.choice(list(self.genotype_constraints.keys()))
            cons = self.genotype_constraints[key]
            if isinstance(cons, tuple):
                new_genotype[key] = np.random.uniform(*cons)
            else:
                new_genotype[key] = cons[np.random.choice(len(cons))]
            new_genotype["_age"] = 0
            new_genotypes.append(new_genotype)
        return new_genotypes

    def _crossover(self, genotype1, genotype2):
        offspring1, offspring2 = {}, {}
        switch = False
        switch_key = np.random.choice(list(self.genotype_constraints.keys()))
        keys = list(self.genotype_constraints.keys())
        np.random.shuffle(keys)
        for key in keys:
            if key == switch_key:
                switch = True
            offspring1[key] = genotype1[key] if switch else genotype2[key]
            offspring2[key] = genotype2[key] if switch else genotype1[key]
        offspring1["_age"] = 0
        offspring2["_age"] = 0
        return [offspring1, offspring2]

    def update(self, f):
        self.epoch += 1
        try:
            n_agents = self.n_agents[self.epoch]
        except (StopIteration, IndexError):
            self.terminated = True
            return

        prev_gen = sorted(
            [(self.population[i], f[i]) for i in range(len(f))], key=lambda x: x[1]
        )
        prev_gen = [v[0] for v in prev_gen if v[0]["_age"] < self._max_age]

        self.population = []
        self.population += [self._random() for _ in range(int(n_agents * self._random_rate))]

        if int(n_agents * self._survivor_rate):
            survivors = [deepcopy(g) for g in prev_gen[-int(n_agents * self._survivor_rate) :]]
            for g in survivors:
                g["_age"] += 1
            self.population += survivors

        mutate_candidates = prev_gen[-int(self._mutate_eligable_pct * len(prev_gen)) :]
        self.population += self._mutate(
            [deepcopy(g) for g in np.random.choice(mutate_candidates, size=int(n_agents * self._mutation_rate))]
        )

        for _ in range(int(n_agents * self._crossover_rate) // 2):
            self.population += self._crossover(
                deepcopy(np.random.choice(prev_gen)),
                deepcopy(np.random.choice(prev_gen)),
            )

        if len(self) < n_agents:
            self.population += self._mutate(np.random.choice(prev_gen, size=n_agents - len(self)))

    def evaluate(self):
        params = [
            (
                self.get_fitness,
                self.cache,
                genotype,
                self._log_fn,
                next(self.multilogger.filename_generator) if self._logging else None,
            )
            for genotype in self.population
        ]
        results = self.backend.distribute(run, params)
        fitnesses = [r[0] for r in results]
        if any(r[1] for r in results):
            self.terminated = True
        return fitnesses
