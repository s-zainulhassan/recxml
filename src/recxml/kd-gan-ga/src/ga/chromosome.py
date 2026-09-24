import uuid
from copy import deepcopy
import numpy as np

class Chromosome:
    def __init__(self, ga_config, initialize=True):
        self.ga_config = ga_config
        self.id = str(uuid.uuid4())
        self.genes = {}
        self.fitness = None
        self.conflict_score = None
        self.metrics = {}
        self.generation = 0
        if initialize:
            self.initialize()

    def initialize(self):
        self.genes = {
            gene: float(np.random.uniform(lower, upper))
            for gene, (lower, upper) in self.ga_config.gene_bounds.items()
        }
        self.fitness = None; self.conflict_score = None; self.metrics = {}

    def clip(self):
        for gene, (lower, upper) in self.ga_config.gene_bounds.items():
            self.genes[gene] = float(np.clip(self.genes[gene], lower, upper))

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return dict(self.genes)

    def summary(self):
        print("Chromosome:", self.id)
        print("Generation:", self.generation)
        print("Genes:", self.genes)
        print("Fitness:", self.fitness)
        print("Conflict:", self.conflict_score)
