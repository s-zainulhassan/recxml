from src.config import config, set_seed
from src.data import load_bibtex_data, make_splits
from src.metrics import compute_inverse_propensity
from src.pipeline import train_teacher

set_seed(42)
config.ensure_directories()

X_train, Y_train, X_test, Y_test = load_bibtex_data(config.data_dir)
train_ds, val_ds, test_ds, Y_train_split = make_splits(
    X_train, Y_train, X_test, Y_test, config.batch_size, seed=42
)
inv_prop = compute_inverse_propensity(Y_train_split)

train_teacher(config, train_ds, val_ds, inv_prop)
print("Teacher training complete.")
