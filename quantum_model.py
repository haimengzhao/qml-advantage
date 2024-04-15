import numpy as np
import pyclifford as pc
from magic_game import check_answer, num_to_bit, bit_to_num
from tqdm import tqdm
from numba import njit

def SWAP(*qubits):
    assert len(qubits) == 2
    layer = pc.circuit.CliffordLayer()
    layer.take(pc.circuit.CNOT(qubits[0], qubits[1]))
    layer.take(pc.circuit.CNOT(qubits[1], qubits[0]))
    layer.take(pc.circuit.CNOT(qubits[0], qubits[1]))
    return layer

def add_noise(state, qubit, noise):
    rand = np.random.rand()
    if rand < noise:
        direction = np.random.randint(0, 3)
        if direction == 0:
            pc.circuit.X(qubit).forward(state)
        elif direction == 1:
            pc.circuit.Y(qubit).forward(state)
        elif direction == 2:
            pc.circuit.Z(qubit).forward(state)
    return state
        
class Quantum_Strategy():
    def __init__(self, n, noise=0):
        self.n = n
        self.n_games = n
        self.n_qubits = 4 * n
        self.noise = noise
        
    def state_prepare(self):
        '''
            2n Bell states 00+11, each on i-th and i+2n-th qubits
        '''
        psi = pc.stabilizer.zero_state(self.n_qubits)
        for i in range(self.n):
            pc.circuit.H(2 * i).forward(psi)
            add_noise(psi, 2 * i, self.noise)
            pc.circuit.CNOT(2 * i, 2 * self.n + 2 * i).forward(psi)
            add_noise(psi, 2 * i, self.noise)
            add_noise(psi, 2 * self.n + 2 * i, self.noise)
            pc.circuit.H(2 * i + 1).forward(psi)
            add_noise(psi, 2 * i + 1, self.noise)
            pc.circuit.CNOT(2 * i + 1, 2 * self.n + 2 * i + 1).forward(psi)
            add_noise(psi, 2 * i + 1, self.noise)
            add_noise(psi, 2 * self.n + 2 * i + 1, self.noise)
        return psi
        # directly construct Bell states, turns out to be slower
        # n = 2 * self.n
        # stab = np.zeros((2*n, 2*n), dtype=int)
        # stab[np.arange(n), np.arange(n)] = 1
        # stab[np.arange(n), np.arange(n) + n] = 1
        # stab[np.arange(n, 2*n), np.arange(n)] = 3
        # stab[np.arange(n, 2*n), np.arange(n) + n] = 3
        # return pc.stabilizer.stabilizer_state(stab)
    
    def get_rotation(self, inp, player, qubits):
        rot = pc.circuit.CliffordLayer()
        if player == 0:
            if inp == 0:
                rot.take(pc.circuit.identity_circuit())
            elif inp == 1:
                rot.take(pc.circuit.H(qubits[0]))
            elif inp == 2:
                rot.take(SWAP(*qubits))
                rot.take(pc.circuit.H(qubits[0]))
            elif inp == 3:
                rot.take(pc.circuit.CNOT(*qubits))
                rot.take(pc.circuit.H(qubits[0]))
        elif player == 1:
            if inp == 0:
                rot.take(pc.circuit.identity_circuit())
            elif inp == 1:
                rot.take(pc.circuit.H(qubits[0]))
                rot.take(pc.circuit.H(qubits[1]))
            elif inp == 2:
                rot.take(SWAP(*qubits))
            elif inp == 3:
                rot.take(pc.circuit.Z(qubits[0]))
                rot.take(pc.circuit.Z(qubits[1]))
                # CZ
                rot.take(pc.circuit.H(qubits[1]))
                rot.take(pc.circuit.CNOT(*qubits))
                rot.take(pc.circuit.H(qubits[1]))
                
                rot.take(pc.circuit.H(qubits[0]))
                rot.take(pc.circuit.H(qubits[1]))
        return rot
    
    def compute(self, inp):
        psi = self.state_prepare()
        for i in range(2 * self.n):
            player = int(i >= self.n)
            rot = self.get_rotation(inp[i], player, [2 * i, 2 * i + 1])
            rot.forward(psi)
            add_noise(psi, 2 * i, self.noise)
            add_noise(psi, 2 * i + 1, self.noise)
        return psi
    
    def sample(self, state, return_log2prob=False):
        measure = pc.circuit.MeasureLayer(*list(range(self.n_qubits)), N=self.n_qubits)
        measure.forward(state)
        sam = measure.result
        sam = ((-sam + 1) / 2).astype(int)
        if return_log2prob:
            return sam, measure.log2prob
        return sam
    
    def compute_and_sample(self, inp, return_log2prob=False):
        state = self.compute(inp)
        return self.sample(state, return_log2prob)
    
    def check_one_answer(self, inp, sample):
        # inp {0, 1, 2, 3}
        inp1 = inp.reshape(2, -1)[0]
        inp2 = inp.reshape(2, -1)[1]
        out1 = sample.reshape(2, -1)[0].reshape(-1, 2)
        out2 = sample.reshape(2, -1)[1].reshape(-1, 2)
        return check_answer(inp1, inp2, out1, out2)
    
    def check_input_output(self, inp, output, flatten=True):
        # inp {0, 1}
        length = len(inp)
        inp = bit_to_num(inp.reshape(-1, 2)).reshape(length, -1)
        inp1 = inp.reshape(length, 2, -1)[:, 0].reshape(-1)
        inp2 = inp.reshape(length, 2, -1)[:, 1].reshape(-1)
        out1 = output.reshape(length, 2, -1)[:, 0].reshape(-1, 2)
        out2 = output.reshape(length, 2, -1)[:, 1].reshape(-1, 2)
        # print(inp1, inp2, out1, out2)
        if flatten:
            return check_answer(inp1, inp2, out1, out2)
        return check_answer(inp1, inp2, out1, out2).reshape(length, -1)
    
    def produce_one_data(self):
        inp = np.random.randint(0, 4, size=2 * self.n)
        sample = self.compute_and_sample(inp)
        return inp, sample
    
    def produce_data(self, n_samples, inp_as_bit=True, save_path=None, progress_bar=True):
        data = {'X': [], 'Y': []}
        iterator = range(n_samples) if not progress_bar else tqdm(range(n_samples))
        for _ in iterator:
            inp, sample = self.produce_one_data()
            if inp_as_bit:
                inp = num_to_bit(inp)
            data['X'].append(inp)
            data['Y'].append(sample)
        data['X'] = np.array(data['X'])
        data['Y'] = np.array(data['Y'])
        if save_path is not None:
            np.savez(save_path, **data)
        return data