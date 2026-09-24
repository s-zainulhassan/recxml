from dataclasses import dataclass, field
from typing import Dict, Tuple
import random
import numpy as np
import tensorflow as tf

@dataclass
class GAConfig:
    ga_seed: int = 42
    fitness_student_epochs: int = 10
    population_size: int = 20
    num_generations: int = 15
    elite_size: int = 2
    ga_patience: int = 8
    tournament_size: int = 3
    crossover_rate: float = 0.80
    mutation_rate: float = 0.20
    mutation_std: float = 0.10
    gene_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "lambda_hard": (0.50, 2.00),
        "lambda_output": (0.10, 1.00),
        "lambda_feature": (0.00, 1.00),
        "lambda_adv": (0.00, 1.00),
    })
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "P@1": 0.25, "P@3": 0.25, "P@5": 0.25,
        "nDCG@5": 0.15, "PSP@5": 0.10,
    })
    conflict_penalty: float = 0.20
    save_ga_history: bool = True
    verbose: bool = True

    def set_random_seed(self):
        random.seed(self.ga_seed); np.random.seed(self.ga_seed); tf.random.set_seed(self.ga_seed)
    def chromosome_length(self): return len(self.gene_bounds)
    def gene_names(self): return list(self.gene_bounds.keys())
    def lower_bounds(self): return [x[0] for x in self.gene_bounds.values()]
    def upper_bounds(self): return [x[1] for x in self.gene_bounds.values()]
