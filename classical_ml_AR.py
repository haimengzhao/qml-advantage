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
    # 0, 1, start=2
    y_train = np.hstack([np.ones((data_size, 1)) * 2, y_train])
    y_train = keras.utils.to_categorical(y_train, 3)
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
    x_inputs = keras.Input(shape=(None, 2))
    y_inputs = keras.Input(shape=(None, 3))
    # stack x and y
    whole_inputs = keras.layers.Concatenate(axis=-1)([x_inputs, y_inputs])
    gru = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True, bias_initializer=keras.initializers.Constant(-1.0))
    gru_outputs, state = gru(whole_inputs)
    final_outputs = keras.layers.Dense(3, activation="softmax")(gru_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([x_inputs, y_inputs], final_outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    )
    batch_size = 1000

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=f"./ckpt/AR_n{n}_l{latent_dim}/{run}.keras", save_best_only=True, monitor="loss"),
        keras.callbacks.EarlyStopping(monitor="loss", patience=500),
        TqdmCallback(verbose=0)
    ]

    model.fit(
        [x_train, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=10000,
        callbacks=callbacks,
        verbose=0,
    )
    return model

def sample_and_predict(n, latent_dim, test_size, max_decoder_seq_length, x_train, x_test):
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model_pred = keras.models.load_model(f"./ckpt/AR_n{n}_l{latent_dim}/{run}.keras")

    x_inputs = model.input[0]  # input_1
    y_inputs = model.input[1]  # input_2
    concat = model.layers[2]  # concatenate_1
    gru = model.layers[3]  # lstm_1
    init_state = keras.Input(shape=(latent_dim,))
    gru_outputs, state = gru(concat([x_inputs, y_inputs]), initial_state=init_state)
    outputs = model.layers[4](gru_outputs)

    model_pred = keras.Model(
        [x_inputs, y_inputs, init_state], [outputs] + [state]
    )

    def decode_sequence(x_input):
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 3))
        # Populate the first character of target sequence with the start character 2.
        target_seq[0, 0, 2] = 1.0
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        init_state = np.zeros((1, latent_dim))
        while not stop_condition:
            output_tokens, new_state = model_pred.predict(
                [x_input[:, [0]]] + [target_seq] + [init_state], verbose=0
            )

            # Sample a token
            sampled_char = np.argmax(output_tokens[0, -1, :])
            decoded_sentence += [sampled_char]

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sentence) == max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 3))
            target_seq[0, 0, sampled_char] = 1.0
            
            init_state = new_state
            x_input = x_input[:, 1:]

        return decoded_sentence
        
    pred = np.zeros((test_size, x_train.shape[1]))
    for seq_index in tqdm(range(test_size)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = x_test[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        try:
            pred[seq_index] = decoded_sentence
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
    # create dir if not exist
    if not os.path.exists("./results"):
        os.makedirs("./results")
    try:
        with open("./results/AR.csv", "r") as f:
            lines = f.readlines()
    except:
        lines = ["n,latent_dim,mean,std\n"]
    with open("./results/AR.csv", "w") as f:
        for line in lines:
            if f"{n},{latent_dim}" in line:
                f.write(line[:-2] + f",{result[0]},{result[1]}\n")
            else:
                f.write(line)
        if not any([f"{n},{latent_dim}" in line for line in lines]):
            f.write(f"{n},{latent_dim},{result[0]},{result[1]}\n")
        
if __name__ == "__main__":
    test_size = 1000
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for latent_dim in [4, 8, 16, 32, 64, 128, 256, 512]:
            for run in range(5, 10):
                print("#"*20 + f"n={n}, latent_dim={latent_dim}, run={run}" + "#"*20)
                x_train, x_test, decoder_input_data, decoder_target_data, max_decoder_seq_length = load_data(n, test_size)
                model = build_model_and_train(n, latent_dim, x_train, decoder_input_data, decoder_target_data)
                result = sample_and_predict(n, latent_dim, test_size, max_decoder_seq_length, x_train, x_test)
                print(result)
                write_result(n, latent_dim, result)