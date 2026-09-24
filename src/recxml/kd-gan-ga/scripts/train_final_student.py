from copy import deepcopy
from src.config import config, set_seed
from src.data import load_bibtex_data, make_splits
from src.metrics import compute_inverse_propensity
from src.models.generator import Generator
from src.models.critic import Critic
from src.models.student import StudentGenerator
from src.pipeline import build_model
from src.trainers.kd import KDTrainer

# Replace these with the GA parameters saved from your optimization run.
BEST_PARAMETERS = {
    "lambda_hard": 1.0,
    "lambda_output": 0.50,
    "lambda_feature": 0.25,
    "lambda_adv": 0.20,
}

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

final_config = deepcopy(config)
for name, value in BEST_PARAMETERS.items():
    setattr(final_config, name, value)

student = build_model(StudentGenerator(final_config), final_config)
if final_config.initial_student_checkpoint.exists():
    student.load_weights(final_config.initial_student_checkpoint)

trainer = KDTrainer(teacher, student, critic, final_config)
trainer.fit(train_ds, val_ds, inv_prop)
print("Final student training complete.")
