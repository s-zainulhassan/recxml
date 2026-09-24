import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.losses import (
    critic_loss, generator_adversarial_loss,
    generator_classification_loss
)
from src.metrics import evaluate_predictions

class WCGANTrainer:
    def __init__(self, generator, critic, config):
        self.generator = generator
        self.critic = critic
        self.config = config
        self.generator_optimizer = Adam(
            learning_rate=config.lr_generator,
            beta_1=config.beta1, beta_2=config.beta2
        )
        self.critic_optimizer = Adam(
            learning_rate=config.lr_critic,
            beta_1=config.beta1, beta_2=config.beta2
        )

    def gradient_penalty(self, documents, real_labels, fake_labels):
        batch_size = tf.shape(documents)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_labels + (1.0 - alpha) * fake_labels
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            scores = self.critic(documents, interpolated)
        gradients = tape.gradient(scores, interpolated)
        gradients = tf.reshape(gradients, [batch_size, -1])
        norm = tf.norm(gradients, axis=1)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def train_critic(self, documents, real_labels):
        batch_size = tf.shape(documents)[0]
        noise = tf.random.normal((batch_size, self.config.noise_dim))
        fake_labels = tf.stop_gradient(self.generator(documents, noise, training=True))
        with tf.GradientTape() as tape:
            real_scores = self.critic(documents, real_labels)
            fake_scores = self.critic(documents, fake_labels)
            wasserstein = critic_loss(real_scores, fake_scores)
            gp = self.gradient_penalty(documents, real_labels, fake_labels)
            total = wasserstein + self.config.lambda_gp * gp
        gradients = tape.gradient(total, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return total

    @tf.function
    def train_generator(self, documents, real_labels):
        batch_size = tf.shape(documents)[0]
        noise = tf.random.normal((batch_size, self.config.noise_dim))
        with tf.GradientTape() as tape:
            fake_labels = self.generator(documents, noise, training=True)
            fake_scores = self.critic(documents, fake_labels)
            adv_loss = generator_adversarial_loss(fake_scores)
            cls_loss = generator_classification_loss(real_labels, fake_labels)
            total = adv_loss + self.config.lambda_cls * cls_loss
        gradients = tape.gradient(total, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return total, adv_loss, cls_loss

    def predict(self, dataset):
        predictions, labels = [], []
        for documents, y in dataset:
            documents = tf.convert_to_tensor(documents, dtype=tf.float32)
            noise = tf.zeros((documents.shape[0], self.config.noise_dim), dtype=tf.float32)
            pred = self.generator(documents, noise, training=False)
            predictions.append(pred.numpy())
            labels.append(y)
        return np.vstack(labels), np.vstack(predictions)

    def evaluate(self, dataset, inverse_propensity):
        y_true, y_pred = self.predict(dataset)
        return evaluate_predictions(y_true, y_pred, inverse_propensity)

    def fit(self, train_dataset, validation_dataset, inverse_propensity):
        best_psp5 = -1.0
        for epoch in range(self.config.epochs):
            d_losses, g_losses, adv_losses, cls_losses = [], [], [], []
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            for documents, labels in train_dataset:
                documents = tf.convert_to_tensor(documents, dtype=tf.float32)
                labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                for _ in range(self.config.critic_steps):
                    d_loss = self.train_critic(documents, labels)
                g_loss, adv_loss, cls_loss = self.train_generator(documents, labels)
                d_losses.append(float(d_loss)); g_losses.append(float(g_loss))
                adv_losses.append(float(adv_loss)); cls_losses.append(float(cls_loss))
            metrics = self.evaluate(validation_dataset, inverse_propensity)
            print(
                f"D={np.mean(d_losses):.4f} | G={np.mean(g_losses):.4f} | "
                f"Adv={np.mean(adv_losses):.4f} | Cls={np.mean(cls_losses):.4f}"
            )
            print("Validation:", {k: round(v, 4) for k, v in metrics.items()})
            if metrics["PSP@5"] > best_psp5:
                best_psp5 = metrics["PSP@5"]
                self.generator.save_weights(self.config.teacher_checkpoint)
                self.critic.save_weights(self.config.critic_checkpoint)
