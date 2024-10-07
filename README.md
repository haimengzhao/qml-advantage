# Entanglement-induced Provable and Robust Quantum Learning Advantages

Welcome to the GitHub repository for the paper "Entanglement-induced Provable and Robust Quantum Learning Advantages". [(arXiv)](https://arxiv.org/abs/2410.03094)

## Abstract

Quantum computing holds the unparalleled potentials to enhance, speed up or innovate machine learning. However, an unambiguous demonstration of quantum learning advantage has not been achieved so far. Here, we rigorously establish a noise-robust, unconditional quantum learning advantage in terms of expressivity, inference speed, and training efficiency, compared to commonly-used classical machine learning models. Our proof is information-theoretic and pinpoints the origin of this advantage: quantum entanglement can be used to reduce the communication required by non-local machine learning tasks. In particular, we design a fully classical task that can be solved with unit accuracy by a quantum model with a constant number of variational parameters using entanglement resources, whereas commonly-used classical models must scale at least linearly with the size of the task to achieve a larger-than-exponentially-small accuracy. We further show that the quantum model can be trained with constant time and a number of samples inversely proportional to the problem size. We prove that this advantage is robust against constant depolarization noise. We show through numerical simulations that even though the classical models can have improved performance as their sizes are increased, they would suffer from overfitting. The constant-versus-linear separation, bolstered by the overfitting problem, makes it possible to demonstrate the  quantum advantage with relatively small system sizes. We demonstrate, through both numerical simulations and trapped-ion experiments on IonQ Aria, the desired quantum-classical learning separation. Our results provide a valuable guide for demonstrating quantum learning advantages in practical applications with current noisy intermediate-scale quantum devices.

## Repository Structure

- `magic_game.py`: Contains the utility functions related to the modified magic square game described in the paper. This includes checking the winning condition of the game, and converting between different representations of the sequences.
- `quantum_model.py`: Implements the optimal quantum model for the magic square translation task. It also includes functions for evaluating the performance of the model and simulating the depolarization noise. It can also be used to generate dataset (with multi-processing support) for the training of classical models. `test.py` tests its correctness.
- `classical_ml_AR.py` and `classical_ml_ED.py`: Scripts for training classical models (autoregressive and encoder-decoder models) via [Keras](https://keras.io) to solve the magic square translation tasks. Training data can be generated using `quantum_model.py` and stored in `./data/`. The results are provided in `./results/` as csv files.
- `*noisy_quantum*`: Reproduces the results on the performance of the quantum model under different noise strength. `noisy_quantum.ipynb` contains a test run, and `production_noisy_quantum.py` implements mass production of the results. The produced results are provided in `noisy_quantum_model.json`.
- `aws.py`: Implements the quantum model on [IonQ Aria](https://ionq.com/quantum-systems/aria) up to 25 qubits via [AWS Braket](https://aws.amazon.com/braket/). The results are stored in `aws.json`. Note that executing this with the default shots number would incur a cost around $30.
- `plot*`: Reproduces the plots in the paper. `utils.py` contains some functions for plotting.

## Prerequisites

`numpy`, `numba`, `matplotlib`, `tqdm`, `orjson`, `SciencePlots` [`PyClifford`](https://github.com/hongyehu/PyClifford), [`keras`](https://keras.io). `line_profiler` for tests.

See [Amazon Braket Doc](https://amazon-braket-sdk-python.readthedocs.io/en/latest/getting-started.html) for requirements related to the AWS tools.

