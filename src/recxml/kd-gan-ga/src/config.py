from pathlib import Path
import random
import numpy as np
import tensorflow as tf

class Config:
    # Dataset
    input_dim = ninput
    label_dim = nlabel
    batch_size = nbatch

    # Teacher WCGAN
    epochs = 100
    lr_generator = 1e-4
    lr_critic = 1e-4
    beta1 = 0.0
    beta2 = 0.9
    critic_steps = 5
    lambda_gp = 10.0
    lambda_cls = 10.0

    # Generator / student
    noise_dim = 128
    hidden_dim = 1024
    embedding_dim = 512
    dropout_rate = 0.10

    # Critic
    critic_hidden_dim = 1024
    critic_embedding_dim = 512
    critic_dropout = 0.20

    # Regularization / prediction
    l2_weight = 1e-5
    threshold = 0.50

    # Student KD
    student_epochs = 100
    student_learning_rate = 1e-4
    student_beta1 = 0.0
    student_beta2 = 0.9
    lambda_hard = 1.0
    lambda_output = 0.50
    lambda_feature = 0.25
    lambda_adv = 0.20
    distillation_temperature = 2.0

    # Paths: change these for your environment.
    data_dir = Path("data")
    checkpoint_dir = Path("checkpoints")
    teacher_checkpoint = checkpoint_dir / "teacher_generator.weights.h5"
    critic_checkpoint = checkpoint_dir / "teacher_critic.weights.h5"
    student_checkpoint = checkpoint_dir / "student_generator.weights.h5"
    initial_student_checkpoint = checkpoint_dir / "initial_student.weights.h5"

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

config = Config()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
