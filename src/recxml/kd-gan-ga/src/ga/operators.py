import random
import numpy as np

class SelectionOperator:
    def __init__(self, ga_config): self.ga_config = ga_config

    def tournament_selection(self, population):
        if len(population) == 0:
            raise ValueError("Population is empty.")
        if any(c.fitness is None for c in population):
            raise ValueError("Population contains unevaluated chromosomes.")
        tournament = random.sample(population.chromosomes, self.ga_config.tournament_size)
        return max(tournament, key=lambda c: c.fitness).copy()

    def select_two_parents(self, population):
        p1 = self.tournament_selection(population)
        p2 = self.tournament_selection(population)
        while p1.id == p2.id:
            p2 = self.tournament_selection(population)
        return p1, p2

class CrossoverOperator:
    def __init__(self, ga_config): self.ga_config = ga_config

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() <= self.ga_config.crossover_rate:
            alpha = random.random()
            for gene in self.ga_config.gene_names():
                p1, p2 = parent1.genes[gene], parent2.genes[gene]
                child1.genes[gene] = alpha*p1 + (1-alpha)*p2
                child2.genes[gene] = (1-alpha)*p1 + alpha*p2
        child1.clip(); child2.clip()
        child1.fitness = child2.fitness = None
        return child1, child2

class MutationOperator:
    def __init__(self, ga_config): self.ga_config = ga_config

    def mutate(self, chromosome):
        offspring = chromosome.copy()
        for gene in self.ga_config.gene_names():
            if random.random() < self.ga_config.mutation_rate:
                lo, hi = self.ga_config.gene_bounds[gene]
                offspring.genes[gene] = float(np.clip(
                    offspring.genes[gene] + np.random.normal(0, self.ga_config.mutation_std),
                    lo, hi
                ))
        offspring.fitness = None
        offspring.clip()
        return offspring
