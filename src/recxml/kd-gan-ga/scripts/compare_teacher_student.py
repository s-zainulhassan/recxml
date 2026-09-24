from src.config import config, set_seed
from src.data import load_bibtex_data, make_splits
from src.metrics import compute_inverse_propensity
from src.models.generator import Generator
from src.models.student import StudentGenerator
from src.pipeline import build_model, evaluate_model

set_seed(42)
config.ensure_directories()
X_train, Y_train, X_test, Y_test = load_bibtex_data(config.data_dir)
_, _, test_ds, Y_train_split = make_splits(
    X_train, Y_train, X_test, Y_test, config.batch_size, seed=42
)
inv_prop = compute_inverse_propensity(Y_train_split)

teacher = build_model(Generator(config), config)
teacher.load_weights(config.teacher_checkpoint)
student = build_model(StudentGenerator(config), config)
student.load_weights(config.student_checkpoint)

teacher_results = evaluate_model(teacher, test_ds, config, inv_prop)
student_results = evaluate_model(student, test_ds, config, inv_prop)

metrics = ["P@1", "P@3", "P@5", "PSP@1", "PSP@3", "PSP@5",
           "nDCG@1", "nDCG@3", "nDCG@5"]
print(f"{'Metric':<12}{'Teacher':>12}{'Student':>12}{'Difference':>15}")
print("-" * 55)
for metric in metrics:
    t, s = teacher_results[metric], student_results[metric]
    print(f"{metric:<12}{t:>12.4f}{s:>12.4f}{s-t:>15.4f}")
