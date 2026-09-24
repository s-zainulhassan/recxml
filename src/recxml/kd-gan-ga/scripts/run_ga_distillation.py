import tensorflow as tf
from src.config import config, set_seed
from src.data import load_bibtex_data, make_splits
from src.metrics import compute_inverse_propensity
from src.models.generator import Generator
from src.models.critic import Critic
from src.pipeline import build_model, run_ga
from src.ga.config import GAConfig

set_seed(42)
config.ensure_directories()
X_train, Y_train, X_test, Y_test = load_bibtex_data(config.data_dir)
train_ds, val_ds, test_ds, Y_train_split = make_splits(
    X_train, Y_train, X_test, Y_test, config.batch_size, seed=42
)
inv_prop = compute_inverse_propensity(Y_train_split)

teacher = build_model(Generator(config), config)
critic = build_model(Critic(config), config)
teacher.load_weights(config.teacher_checkpoint)
critic.load_weights(config.critic_checkpoint)

best = run_ga(config, teacher, critic, train_ds, val_ds, inv_prop)
print("Best GA parameters:", best.to_dict())
