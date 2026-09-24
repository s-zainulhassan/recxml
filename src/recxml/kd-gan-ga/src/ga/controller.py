from src.ga.population import Population
from src.ga.fitness import FitnessEvaluator
from src.ga.operators import SelectionOperator, CrossoverOperator, MutationOperator

class GAController:
    def __init__(self, config, ga_config, teacher_generator, teacher_critic,
                 train_dataset, val_dataset, inverse_propensity):
        self.config = config; self.ga_config = ga_config
        self.train_dataset = train_dataset; self.val_dataset = val_dataset
        self.inverse_propensity = inverse_propensity
        self.teacher_generator = teacher_generator; self.teacher_critic = teacher_critic
        self.population = Population(ga_config)
        self.fitness_evaluator = FitnessEvaluator(
            config, ga_config, teacher_generator, teacher_critic,
            train_dataset, val_dataset, inverse_propensity
        )
        self.selection = SelectionOperator(ga_config)
        self.crossover = CrossoverOperator(ga_config)
        self.mutation = MutationOperator(ga_config)
        self.best_chromosome = None
        self.current_generation = 0

    def initialize_population(self):
        self.population.initialize()
        self.current_generation = 0

    def evaluate_population(self):
        for chromosome in self.population:
            self.fitness_evaluator.evaluate(chromosome)
        best = self.population.best()
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best.copy()
        return best

    def create_next_generation(self):
        new_population = Population(self.ga_config)
        new_population.generation = self.current_generation + 1
        for elite in self.population.elites():
            elite.generation = self.current_generation + 1
            new_population.add(elite)
        while len(new_population) < self.ga_config.population_size:
            p1, p2 = self.selection.select_two_parents(self.population)
            c1, c2 = self.crossover.crossover(p1, p2)
            new_population.add(self.mutation.mutate(c1))
            if len(new_population) < self.ga_config.population_size:
                new_population.add(self.mutation.mutate(c2))
        return new_population

    def run(self):
        self.initialize_population()
        self.evaluate_population()
        patience = 0
        best_seen = self.best_chromosome.fitness
        for _ in range(self.ga_config.num_generations):
            self.population = self.create_next_generation()
            self.current_generation += 1
            self.evaluate_population()
            if self.best_chromosome.fitness > best_seen:
                best_seen = self.best_chromosome.fitness
                patience = 0
            else:
                patience += 1
            if patience >= self.ga_config.ga_patience:
                break
        return self.best_chromosome

    def best_parameters(self):
        return None if self.best_chromosome is None else self.best_chromosome.to_dict()

    def optimization_summary(self):
        if self.best_chromosome is None:
            print("No optimization has been performed.")
            return
        print("Best fitness:", self.best_chromosome.fitness)
        print("Conflict:", self.best_chromosome.conflict_score)
        print("Parameters:", self.best_chromosome.genes)
