import numpy as np
from src.ga.chromosome import Chromosome

class Population:
    def __init__(self, ga_config):
        self.ga_config = ga_config
        self.chromosomes = []
        self.generation = 0

    def initialize(self):
        self.chromosomes = [
            Chromosome(self.ga_config)
            for _ in range(self.ga_config.population_size)
        ]

    def add(self, chromosome):
        self.chromosomes.append(chromosome)

    def __len__(self): return len(self.chromosomes)
    def __iter__(self): return iter(self.chromosomes)
    def __getitem__(self, i): return self.chromosomes[i]

    def sort_by_fitness(self, reverse=True):
        self.chromosomes.sort(
            key=lambda c: -np.inf if c.fitness is None else c.fitness,
            reverse=reverse
        )

    def best(self):
        self.sort_by_fitness()
        return self.chromosomes[0]

    def elites(self):
        self.sort_by_fitness()
        return [c.copy() for c in self.chromosomes[:self.ga_config.elite_size]]

    def statistics(self):
        values = [c.fitness for c in self.chromosomes if c.fitness is not None]
        return {"best": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None}
