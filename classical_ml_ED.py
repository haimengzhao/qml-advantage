import numpy as np
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import keras
from quantum_model import Quantum_Strategy
from tqdm import tqdm
from tqdm.keras import TqdmCallback

threshold = 0.95

def load_data(n, test_size):
    data = np.load(f"./data/data_{n}.npz")
    data_size = len(data['X'])
    x_train = data['X']; y_train = data['Y']
    x_test = np.random.randint(0, 2, (test_size, x_train.shape[1]))
    # one-hot
    x_train = keras.utils.to_categorical(x_train, 2)
    x_test = keras.utils.to_categorical(x_test, 2)
    # 0, 1, start=2, end=3
    y_train = np.hstack([np.ones((data_size, 1)) * 2, y_train, np.ones((data_size, 1)) * 3])
    y_train = keras.utils.to_categorical(y_train, 4)
    decoder_input_data = y_train[:, :-1]
    decoder_target_data = y_train[:, 1:]
    print("n=", n)
    print("n_qubits=", 4 * n)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    max_decoder_seq_length = decoder_input_data.shape[1]
    return x_train, x_test, decoder_input_data, decoder_target_data, max_decoder_seq_length

def build_model_and_train(n, latent_dim, x_train, decoder_input_data, decoder_target_data, verbose=0):
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, 2))
    encoder = keras.layers.GRU(latent_dim, return_state=True, bias_initializer=keras.initializers.Constant(-1.0))
    encoder_outputs, state_h = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = state_h

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, 4))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True, bias_initializer=keras.initializers.Constant(-1.0))
    decoder_outputs, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(4, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=5e-3),
    )
    batch_size = 1000

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=f"./ckpt/ED_n{n}_l{latent_dim}/{run}.keras", save_best_only=True, monitor="loss"),
        keras.callbacks.EarlyStopping(monitor="loss", patience=500),
        TqdmCallback(verbose=0)
    ]

    model.fit(
        [x_train, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=10000,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model

def sample_and_predict(n, latent_dim, test_size, max_decoder_seq_length, x_train, x_test):
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model(f"./ckpt/ED_n{n}_l{latent_dim}/{run}.keras")

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc = model.layers[2].output  # lstm_1
    encoder_states = state_h_enc
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = decoder_state_input_h
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = state_h_dec
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + [decoder_states_inputs], [decoder_outputs] + [decoder_states]
    )

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq, verbose=0)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 4))
        # Populate the first character of target sequence with the start character 2.
        target_seq[0, 0, 2] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h = decoder_model.predict(
                [target_seq] + [states_value], verbose=0
            )

            # Sample a token
            sampled_char = np.argmax(output_tokens[0, -1, :])
            decoded_sentence += [sampled_char]


            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == 3 or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 4))
            target_seq[0, 0, sampled_char] = 1.0

            # Update states
            states_value = h
        return decoded_sentence
    
    pred = np.zeros((test_size, x_train.shape[1]))
    for seq_index in tqdm(range(test_size)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = x_test[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        try:
            pred[seq_index] = decoded_sentence[:-1]
        except:
            print('decode stop early: ', decoded_sentence)
        
    qs = Quantum_Strategy(n)
    results = qs.check_input_output(np.argmax(x_test[:test_size], axis=-1), pred, flatten=False)
    check = np.sum(results, axis=-1) > threshold * n
    mean = np.mean(check)
    std = 1.96 * np.sqrt(mean * (1 - mean) / len(check))
    return mean, std

def write_result(n, latent_dim, result):
    # add results to csv file
    # if n, latent_dim not in csv, add new row
    # if n, latent_dim in csv, append mean and std at the end of the row
    if not os.path.exists("./results"):
        os.makedirs("./results")
    try:
        with open("./results/ED_1.csv", "r") as f:
            lines = f.readlines()
    except:
        lines = ["n,latent_dim,mean,std\n"]
    with open("./results/ED_1.csv", "w") as f:
        for line in lines:
            if f"{n},{latent_dim}" in line:
                f.write(line[:-2] + f",{result[0]},{result[1]}\n")
            else:
                f.write(line)
        if not any([f"{n},{latent_dim}" in line for line in lines]):
            f.write(f"{n},{latent_dim},{result[0]},{result[1]}\n")
        
if __name__ == "__main__":
    test_size = 1000
    # resume = None
    resume = (2, 30)
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for latent_dim in np.floor(2**np.linspace(2, 9, 20)).astype(int).tolist():
            # resume
            if resume is not None:
                if n < resume[0] or (n == resume[0] and latent_dim < resume[1]):
                    continue
            for run in range(10):
                print("#"*20 + f"n={n}, latent_dim={latent_dim}, run={run}" + "#"*20)
                x_train, x_test, decoder_input_data, decoder_target_data, max_decoder_seq_length = load_data(n, test_size)
                model = build_model_and_train(n, latent_dim, x_train, decoder_input_data, decoder_target_data)
                result = sample_and_predict(n, latent_dim, test_size, max_decoder_seq_length, x_train, x_test)
                print(result)
                write_result(n, latent_dim, result)