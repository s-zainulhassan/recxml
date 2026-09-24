import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.losses import student_total_loss
from src.metrics import evaluate_predictions

class KDTrainer:
    def __init__(self, teacher, student, critic, config):
        self.teacher = teacher
        self.student = student
        self.critic = critic
        self.config = config
        self.teacher.trainable = False
        self.critic.trainable = False
        self.student_optimizer = Adam(
            learning_rate=config.student_learning_rate,
            beta_1=config.student_beta1, beta_2=config.student_beta2
        )

    @tf.function
    def train_student(self, documents, real_labels):
        batch_size = tf.shape(documents)[0]
        noise = tf.random.normal((batch_size, self.config.noise_dim))
        teacher_predictions = self.teacher(documents, noise, training=False)
        teacher_embedding = self.teacher.embedding
        with tf.GradientTape() as tape:
            student_predictions = self.student(documents, noise, training=True)
            student_embedding = self.student.projected_embedding
            critic_scores = self.critic(documents, student_predictions, training=False)
            losses = student_total_loss(
                real_labels, teacher_predictions, student_predictions,
                teacher_embedding, student_embedding, critic_scores, self.config
            )
        gradients = tape.gradient(losses[0], self.student.trainable_variables)
        self.student_optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        return losses

    def predict(self, dataset):
        predictions, labels = [], []
        for documents, y in dataset:
            documents = tf.convert_to_tensor(documents, dtype=tf.float32)
            noise = tf.zeros((documents.shape[0], self.config.noise_dim), dtype=tf.float32)
            pred = self.student(documents, noise, training=False)
            predictions.append(pred.numpy()); labels.append(y)
        return np.vstack(labels), np.vstack(predictions)

    def evaluate(self, dataset, inverse_propensity):
        y_true, y_pred = self.predict(dataset)
        return evaluate_predictions(y_true, y_pred, inverse_propensity)

    def fit(self, train_dataset, validation_dataset, inverse_propensity):
        best_psp5 = -1.0
        best_metrics = None
        for epoch in range(self.config.student_epochs):
            losses = [[], [], [], [], []]
            print(f"Epoch {epoch + 1}/{self.config.student_epochs}")
            for documents, labels in train_dataset:
                documents = tf.convert_to_tensor(documents, dtype=tf.float32)
                labels = tf.convert_to_tensor(labels, dtype=tf.float32)
                batch_losses = self.train_student(documents, labels)
                for j, value in enumerate(batch_losses):
                    losses[j].append(float(value))
            metrics = self.evaluate(validation_dataset, inverse_propensity)
            print("Losses:", [round(float(np.mean(x)), 4) for x in losses])
            print("Validation:", {k: round(v, 4) for k, v in metrics.items()})
            if metrics["PSP@5"] > best_psp5:
                best_psp5 = metrics["PSP@5"]
                best_metrics = metrics.copy()
                self.student.save_weights(self.config.student_checkpoint)
        return {
            "metrics": best_metrics if best_metrics is not None else metrics,
            "losses": {
                "total": float(np.mean(losses[0])),
                "hard": float(np.mean(losses[1])),
                "output": float(np.mean(losses[2])),
                "feature": float(np.mean(losses[3])),
                "adv": float(np.mean(losses[4])),
            },
        }
