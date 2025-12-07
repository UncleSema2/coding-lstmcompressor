import torch
import torch.nn as nn
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

# Set deterministic operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

# The batch size for training
batch_size = 256
# The sequence length for training
seq_length = 16
# The number of units in each LSTM layer
rnn_units = 256
# The number of LSTM layers
num_layers = 2
# The size of the embedding layer
embedding_size = 256
# The initial learning rate for optimizer
start_learning_rate = 0.005
# The final learning rate for optimizer
end_learning_rate = 0.001
# The mode for the program, "compress", "decompress", "both"
mode = 'both'

path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = embedding_size if i == 0 else embedding_size + rnn_units
            lstm = nn.LSTM(input_size, rnn_units, batch_first=True)
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
            self.lstm_layers.append(lstm)

        output_size = rnn_units * num_layers if num_layers > 1 else rnn_units
        self.dense = nn.Linear(output_size, vocab_size)
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, inputs):
        x = inputs[0]
        states = inputs[1:]

        embedded = self.embedding(x)

        # Skip connections will be used to connect each LSTM layer output to the final
        # output layer. Each LSTM layer will get as input both the original input and
        # the output of the previous layer.
        skip_connections = []
        new_states = []

        h0 = states[0].unsqueeze(0).contiguous()  # [1, batch_size, rnn_units]
        c0 = states[1].unsqueeze(0).contiguous()  # [1, batch_size, rnn_units]
        lstm_out, (hn, cn) = self.lstm_layers[0](embedded, (h0, c0))
        skip_connections.append(lstm_out)
        new_states.extend([hn.squeeze(0), cn.squeeze(0)])

        for i in range(1, num_layers):
            layer_input = torch.cat([embedded, skip_connections[-1]], dim=-1)
            h0 = states[i*2].unsqueeze(0).contiguous()  # [1, batch_size, rnn_units]
            c0 = states[i*2+1].unsqueeze(0).contiguous()  # [1, batch_size, rnn_units]
            lstm_out, (hn, cn) = self.lstm_layers[i](layer_input, (h0, c0))
            skip_connections.append(lstm_out)
            new_states.extend([hn.squeeze(0), cn.squeeze(0)])

        # The dense output layer only needs to be computed for the last timestep, so
        # we can discard the earlier outputs.
        last_timestep = []
        for i in range(num_layers):
            last_timestep.append(skip_connections[i][:, -1, :])  # [batch_size, rnn_units]
        if num_layers == 1:
            layer_input = last_timestep[0]
        else:
            layer_input = torch.cat(last_timestep, dim=-1)  # [batch_size, rnn_units * num_layers]

        logits = self.dense(layer_input)
        predictions = torch.softmax(logits, dim=-1)

        outputs = [predictions] + new_states
        return outputs


def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress,
          data, states):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the *first* batch.
        seq_input: Tensor, containing the last seq_length inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.
        states: List containing state information for the layers of the model.

    Returns:
        seq_input: Tensor, containing the last seq_length inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    cross_entropy = denom = 0
    split = math.ceil(length / batch_size)

    seq_input = seq_input

    model_inputs = [seq_input] + [state.detach() for state in states[0]]

    model_outputs = model(model_inputs)
    predictions = model_outputs[0]
    new_states = model_outputs[1:]

    states[0] = new_states

    p = predictions.detach().cpu().numpy()
    symbols = []
    # When the last batch reaches the end of the file, we start giving it "0"
    # as input. We use a mask to prevent this from influencing the gradients.
    mask = []
    # Go over each batch to run the arithmetic coding and prepare the next
    # input.
    for i in range(batch_size):
        # The "10000000" is used to convert floats into large integers (since
        # the arithmetic coder works on integers).
        freq = np.cumsum(p[i] * 10000000 + 1)
        index = pos + 1 + i * split
        symbol = get_symbol(index, length, freq, coder, compress, data)
        symbols.append(symbol)
        if index < length:
            prob = p[i][symbol]
            if prob <= 0:
                # Set a small value to avoid error with log2.
                prob = 0.000001
            cross_entropy += math.log2(prob)
            denom += 1
            mask.append(1.0)
        else:
            mask.append(0.0)

    symbols_tensor = torch.tensor(symbols, dtype=torch.long)

    target_probs = torch.nn.functional.one_hot(symbols_tensor, vocab_size).float()
    loss = -torch.sum(target_probs * torch.log(predictions + 1e-7), dim=-1, keepdim=True)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(1)
    loss = loss * mask_tensor
    loss = loss.sum() / mask_tensor.sum()

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 4)

    optimizer.step()

    seq_input = seq_input[:, 1:]
    symbols_tensor = symbols_tensor.unsqueeze(1)
    seq_input = torch.cat([seq_input, symbols_tensor], dim=1)

    return (seq_input, cross_entropy, denom)


def reset_seed():
    """Initializes various random seeds to help with determinism."""
    SEED = 1234
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def process(compress, length, vocab_size, coder, data):
    """This runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()
    reset_seed()
    model = LSTMModel(vocab_size=vocab_size)
    print(model)

    # Try to split the file into equal size pieces for the different batches. The
    # last batch may have fewer characters if the file can't be split equally.
    split = math.ceil(length / batch_size)

    def lr_lambda(epoch):
        return 1.0 - min(1.0, epoch / split) * (1.0 - end_learning_rate / start_learning_rate)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=start_learning_rate,
        betas=(0.0, 0.9999),
        eps=1e-5
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Use a uniform distribution for predicting the first batch of symbols. The
    # "10000000" is used to convert floats into large integers (since the
    # arithmetic coder works on integers).
    freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)
    # Construct the first set of input characters for training.
    symbols = []
    for i in range(batch_size):
        symbols.append(get_symbol(i*split, length, freq, coder, compress, data))
    # Replicate the input tensor seq_length times, to match the input format.
    symbols_tensor = torch.tensor(symbols, dtype=torch.long)
    seq_input = symbols_tensor.unsqueeze(1).repeat(1, seq_length)
    pos = cross_entropy = denom = 0
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'
    # This will keep track of layer states. Initialize them to zeros.
    states = []
    layer_states = []
    for _ in range(num_layers):
        layer_states.append(torch.zeros(batch_size, rnn_units))
        layer_states.append(torch.zeros(batch_size, rnn_units))
    states.append(layer_states)

    # Keep repeating the training step until we get to the end of the file.
    while pos < split:
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model,
                                 optimizer, compress, data, states)
        cross_entropy += ce
        denom += d
        pos += 1
        scheduler.step()
        if pos % 5 == 0:
            percentage = 100 * pos / split
            if percentage >= 100:
                continue
            print(template.format(percentage, -cross_entropy / denom, time.time() - start))
    if compress:
        coder.finish()
    print(template.format(100, -cross_entropy / length, time.time() - start))


def compession():
    # int_list will contain the characters of the file.
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}
    for idx, c in enumerate(text):
        int_list.append(char2idx[c])

    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size/8) * 8
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        # Write 256 bits to the compressed file header to keep track of the vocabulary.
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)


def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        # Read the original file size from the header.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        # Create a list to store the file characters.
        output = [0] * length
        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        # Round up to a multiple of 8 to improve performance.
        vocab_size = math.ceil(vocab_size/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        # The decompressed data is stored in the "output" list. We can now write the
        # data to file (based on the type of preprocessing used).

        # Convert indexes back to the original characters.
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file)/os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()
