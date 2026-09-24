import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz

class XMLCDataset:
    def __init__(self, X, Y, batch_size=16, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.on_epoch_end()
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            idx = self.indices[start:end]
            X_batch = self.X[idx]
            Y_batch = self.Y[idx]
            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()
            if sp.issparse(Y_batch):
                Y_batch = Y_batch.toarray()
            yield X_batch.astype(np.float32), Y_batch.astype(np.float32)

def load_bibtex_data(data_dir):
    data_dir = str(data_dir)
    return (
        load_npz(f"{data_dir}/X_train.npz"),
        load_npz(f"{data_dir}/Y_train.npz"),
        load_npz(f"{data_dir}/X_test.npz"),
        load_npz(f"{data_dir}/Y_test.npz"),
    )

def make_splits(X_train, Y_train, X_test, Y_test, batch_size, seed=42):
    from sklearn.model_selection import train_test_split
    indices = np.arange(X_train.shape[0])
    train_idx, val_idx = train_test_split(
        indices, test_size=0.10, random_state=seed, shuffle=True
    )
    train_dataset = XMLCDataset(
        X_train[train_idx], Y_train[train_idx],
        batch_size=batch_size, shuffle=True
    )
    validation_dataset = XMLCDataset(
        X_train[val_idx], Y_train[val_idx],
        batch_size=batch_size, shuffle=False
    )
    test_dataset = XMLCDataset(
        X_test, Y_test, batch_size=batch_size, shuffle=False
    )
    return train_dataset, validation_dataset, test_dataset, Y_train[train_idx]
