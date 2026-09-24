from copy import deepcopy
import tensorflow as tf
from src.models.student import StudentGenerator
from src.trainers.kd import KDTrainer

class FitnessEvaluator:
    def __init__(self, config, ga_config, teacher_generator, teacher_critic,
                 train_dataset, val_dataset, inverse_propensity):
        self.config = config; self.ga_config = ga_config
        self.teacher_generator = teacher_generator; self.teacher_critic = teacher_critic
        self.train_dataset = train_dataset; self.val_dataset = val_dataset
        self.inverse_propensity = inverse_propensity

    def create_temp_config(self, chromosome):
        cfg = deepcopy(self.config)
        for gene, value in chromosome.genes.items():
            setattr(cfg, gene, value)
        cfg.student_epochs = self.ga_config.fitness_student_epochs
        return cfg

    def create_trainer(self, temp_config):
        student = StudentGenerator(temp_config)
        dummy_documents = tf.zeros((1, temp_config.input_dim), tf.float32)
        dummy_noise = tf.zeros((1, temp_config.noise_dim), tf.float32)
        student(dummy_documents, dummy_noise, training=False)
        student.load_weights(temp_config.initial_student_checkpoint)
        return KDTrainer(self.teacher_generator, student, self.teacher_critic, temp_config)

    def compute_conflict(self, losses):
        required = ["hard", "output", "feature", "adv"]
        for key in required:
            if key not in losses: raise KeyError(f"Missing loss '{key}'.")
        return float(sum(abs(float(losses["adv"]) - float(losses[k])) for k in required[:-1]) / 3.0)

    def compute_fitness(self, metrics, conflict):
        performance = sum(
            weight * float(metrics[metric])
            for metric, weight in self.ga_config.fitness_weights.items()
        )
        return float(performance - self.ga_config.conflict_penalty * float(conflict))

    def evaluate(self, chromosome):
        chromosome.fitness = chromosome.conflict_score = None
        chromosome.metrics = {}
        temp_config = self.create_temp_config(chromosome)
        trainer = self.create_trainer(temp_config)
        results = trainer.fit(self.train_dataset, self.val_dataset, self.inverse_propensity)
        chromosome.conflict_score = self.compute_conflict(results["losses"])
        chromosome.metrics = results["metrics"]
        chromosome.fitness = self.compute_fitness(chromosome.metrics, chromosome.conflict_score)
        return chromosome
