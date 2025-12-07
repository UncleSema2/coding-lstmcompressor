import torch
import torch.nn as nn
import numpy as np
import random
import time
import math
import contextlib
import os
import optuna
from optuna.trial import Trial
import io
import sqlite3
from datetime import datetime

from ArithmeticCoder import (
    ArithmeticEncoder,
    ArithmeticDecoder,
    BitOutputStream,
    BitInputStream,
)

import importlib.util

spec = importlib.util.spec_from_file_location("rnn_module", "7.0.optuna-prep.py")
rnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnn_module)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

path_to_file = "data/enwik5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TIME_CONSTRAINT = 5.545

DB_FILE = "optuna_results.db"


def process_with_return(compress, length, vocab_size, coder, data):
    rnn_module.reset_seed()
    model = rnn_module.RNNModel(
        vocab_size=vocab_size, cell_type=rnn_module.rnn_type
    ).to(device)

    split = math.ceil(length / rnn_module.batch_size)

    def lr_lambda(epoch):
        return 1.0 - min(1.0, epoch / split) * (
            1.0 - rnn_module.end_learning_rate / rnn_module.start_learning_rate
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=rnn_module.start_learning_rate,
        betas=(0.0, 0.9999),
        eps=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)
    symbols = []
    for i in range(rnn_module.batch_size):
        symbols.append(
            rnn_module.get_symbol(i * split, length, freq, coder, compress, data)
        )

    symbols_tensor = torch.tensor(symbols, dtype=torch.long, device=device)
    seq_input = symbols_tensor.unsqueeze(1).repeat(1, rnn_module.seq_length)

    pos = cross_entropy = denom = 0
    states = []
    layer_states = []
    for j in range(rnn_module.num_layers):
        layer_states.append(
            torch.zeros(rnn_module.batch_size, rnn_module.rnn_units, device=device)
        )
        if rnn_module.rnn_type.lower() == "lstm":
            layer_states.append(
                torch.zeros(rnn_module.batch_size, rnn_module.rnn_units, device=device)
            )
    states.append(layer_states)

    while pos < split:
        seq_input, ce, d = rnn_module.train(
            pos,
            seq_input,
            length,
            vocab_size,
            coder,
            model,
            optimizer,
            compress,
            data,
            states,
        )
        cross_entropy += ce
        denom += d
        pos += 1
        scheduler.step()

    if compress:
        coder.finish()

    return -cross_entropy / length if length > 0 else 0.0


def run_compression_decompression(
    rnn_type,
    batch_size,
    seq_length,
    rnn_units,
    num_layers,
    embedding_size,
    start_learning_rate,
    end_learning_rate,
    char2idx,
    int_list,
    vocab_size,
    length,
):
    compressed_out = io.BytesIO()
    compressed_out.write(length.to_bytes(5, byteorder="big", signed=False))

    bitout = BitOutputStream(compressed_out)
    for i in range(256):
        if i in char2idx:
            bitout.write(1)
        else:
            bitout.write(0)

    enc = ArithmeticEncoder(32, bitout)

    compression_start = time.time()
    compression_ce = process_with_return(True, length, vocab_size, enc, int_list)
    compression_time = time.time() - compression_start

    compressed_out.seek(0)
    length_read = int.from_bytes(compressed_out.read(5), byteorder="big")
    output = [0] * length_read

    bitin = BitInputStream(compressed_out)
    vocab = []
    for i in range(256):
        if bitin.read():
            vocab.append(i)

    vocab_size_dec = len(vocab)
    vocab_size_dec = math.ceil(vocab_size_dec / 8) * 8

    dec = ArithmeticDecoder(32, bitin)

    decompression_start = time.time()
    decompression_ce = process_with_return(
        False, length_read, vocab_size_dec, dec, output
    )
    decompression_time = time.time() - decompression_start

    total_time = compression_time + decompression_time

    return compression_ce, total_time


def evaluate_hyperparameters(trial: Trial):
    rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru", "rnn"])
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    seq_length = trial.suggest_int("seq_length", 4, 32)
    rnn_units = trial.suggest_int("rnn_units", 8, 256)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    embedding_size = trial.suggest_int("embedding_size", 8, 256)
    start_learning_rate = trial.suggest_loguniform("start_learning_rate", 1e-6, 1e-1)
    end_learning_rate = trial.suggest_loguniform("end_learning_rate", 1e-6, 1e-1)

    rnn_module.rnn_type = rnn_type
    rnn_module.batch_size = batch_size
    rnn_module.seq_length = seq_length
    rnn_module.rnn_units = rnn_units
    rnn_module.num_layers = num_layers
    rnn_module.embedding_size = embedding_size
    rnn_module.start_learning_rate = start_learning_rate
    rnn_module.end_learning_rate = end_learning_rate

    int_list = []
    text = open(path_to_file, "rb").read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    char2idx = {u: i for i, u in enumerate(vocab)}
    for idx, c in enumerate(text):
        int_list.append(char2idx[c])

    vocab_size = math.ceil(vocab_size / 8) * 8
    length = len(int_list)

    cross_entropy, total_time = run_compression_decompression(
        rnn_type,
        batch_size,
        seq_length,
        rnn_units,
        num_layers,
        embedding_size,
        start_learning_rate,
        end_learning_rate,
        char2idx,
        int_list,
        vocab_size,
        length,
    )

    trial.set_user_attr("total_time", total_time)
    trial.set_user_attr("constraint_satisfied", total_time <= TIME_CONSTRAINT)

    if total_time > TIME_CONSTRAINT:
        print(
            f"Trial {trial.number}: TIME EXCEEDED {total_time:.2f}s > {TIME_CONSTRAINT}s"
        )
        return float("inf")

    print(f"Trial {trial.number}: CE={cross_entropy:.4f}, Time={total_time:.2f}s")
    return cross_entropy


def main():
    start_time = time.time()

    storage_url = f"sqlite:///{DB_FILE}"
    study = optuna.create_study(
        direction="minimize",
        study_name="rnn_compression_optimization",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1000,
            interval_steps=50,
        ),
    )

    print("Starting hyperparameter optimization with Optuna...")
    print(f"Time constraint: {TIME_CONSTRAINT} seconds (compression + decompression)")
    print(f"Testing RNN types: LSTM, GRU, RNN")
    print("-" * 80)

    n_trials = 500
    study.optimize(evaluate_hyperparameters, n_trials=n_trials, timeout=None)


if __name__ == "__main__":
    main()
