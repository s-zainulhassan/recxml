import argparse
from src.config import config, set_seed
from src.data import load_bibtex_data, make_splits
from src.metrics import compute_inverse_propensity
from src.models.generator import Generator
from src.models.student import StudentGenerator
from src.pipeline import build_model, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["teacher", "student"], required=True)
args = parser.parse_args()

set_seed(42)
config.ensure_directories()
X_train, Y_train, X_test, Y_test = load_bibtex_data(config.data_dir)
_, _, test_ds, Y_train_split = make_splits(
    X_train, Y_train, X_test, Y_test, config.batch_size, seed=42
)
inv_prop = compute_inverse_propensity(Y_train_split)

if args.model == "teacher":
    model = build_model(Generator(config), config)
    model.load_weights(config.teacher_checkpoint)
else:
    model = build_model(StudentGenerator(config), config)
    model.load_weights(config.student_checkpoint)

results = evaluate_model(model, test_ds, config, inv_prop)
print(results)
