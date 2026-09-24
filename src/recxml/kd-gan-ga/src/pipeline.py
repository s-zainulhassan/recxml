import numpy as np
import tensorflow as tf
from src.models.generator import Generator
from src.models.critic import Critic
from src.models.student import StudentGenerator
from src.trainers.wcgan import WCGANTrainer
from src.trainers.kd import KDTrainer
from src.metrics import compute_inverse_propensity, evaluate_predictions
from src.ga.config import GAConfig
from src.ga.controller import GAController

def build_model(model, config):
    dummy_documents = tf.zeros((1, config.input_dim), tf.float32)
    dummy_noise = tf.zeros((1, config.noise_dim), tf.float32)
    if isinstance(model, Critic):
        dummy_labels = tf.zeros((1, config.label_dim), tf.float32)
        model(dummy_documents, dummy_labels, training=False)
    else:
        model(dummy_documents, dummy_noise, training=False)
    return model

def train_teacher(config, train_dataset, validation_dataset, inverse_propensity):
    generator = build_model(Generator(config), config)
    critic = build_model(Critic(config), config)
    trainer = WCGANTrainer(generator, critic, config)
    trainer.fit(train_dataset, validation_dataset, inverse_propensity)
    return generator, critic

def train_student(config, teacher, critic, train_dataset, validation_dataset, inverse_propensity):
    student = build_model(StudentGenerator(config), config)
    trainer = KDTrainer(teacher, student, critic, config)
    return trainer, trainer.fit(train_dataset, validation_dataset, inverse_propensity)

def run_ga(config, teacher, critic, train_dataset, validation_dataset, inverse_propensity):
    ga_config = GAConfig()
    ga_config.set_random_seed()
    controller = GAController(
        config, ga_config, teacher, critic,
        train_dataset, validation_dataset, inverse_propensity
    )
    return controller.run()

def predict_model(model, dataset, config):
    y_true, y_pred = [], []
    for documents, labels in dataset:
        documents = tf.convert_to_tensor(documents, tf.float32)
        noise = tf.zeros((documents.shape[0], config.noise_dim), tf.float32)
        y_pred.append(model(documents, noise, training=False).numpy())
        y_true.append(labels)
    return np.vstack(y_true), np.vstack(y_pred)

def evaluate_model(model, dataset, config, inverse_propensity):
    y_true, y_pred = predict_model(model, dataset, config)
    return evaluate_predictions(y_true, y_pred, inverse_propensity)
