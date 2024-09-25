import boto3
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from braket.tracking import Tracker

import numpy as np
import orjson
from tqdm import tqdm
from magic_game import check_answer, num_to_bit, bit_to_num

t = Tracker()

# get the account ID
aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
# the name of the bucket
my_bucket = "amazon-braket-us-east-1-241533162288"
# the name of the folder in the bucket
my_prefix = "simulation-output"
s3_folder = (my_bucket, my_prefix)

n = 6
n_qubits = 4 * n
n_inp = 10
n_shot = 100
device = AwsDevice('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1')

def state_prepare():
    '''
        2n Bell states 00+11, each on i-th and i+2n-th qubits
    '''
    circ = Circuit()
    for i in range(n):
        circ.h(2 * i)
        circ.cnot(2 * i, 2 * n + 2 * i)
        circ.h(2 * i + 1)
        circ.cnot(2 * i + 1, 2 * n + 2 * i + 1)
    return circ
    
def get_rotation(circ, inp, player, qubits):
    if player == 0:
        if inp == 0:
            pass
        elif inp == 1:
            circ.h(qubits[0])
        elif inp == 2:
            circ.swap(qubits[0], qubits[1])
            circ.h(qubits[0])
        elif inp == 3:
            circ.cnot(qubits[0], qubits[1])
            circ.h(qubits[0])
    elif player == 1:
        if inp == 0:
            pass
        elif inp == 1:
            circ.h(qubits[0])
            circ.h(qubits[1])
        elif inp == 2:
            circ.swap(qubits[0], qubits[1])
        elif inp == 3:
            circ.z(qubits[0])
            circ.z(qubits[1])
            # CZ
            circ.h(qubits[1])
            circ.cnot(qubits[0], qubits[1])
            circ.h(qubits[1])
            
            circ.h(qubits[0])
            circ.h(qubits[1])
    return circ

def compute(inp):
    circ = state_prepare()
    for i in range(2 * n):
        player = int(i >= n)
        circ = get_rotation(circ, inp[i], player, [2 * i, 2 * i + 1])
    return circ

def check_one_answer(inp, sample):
        # inp {0, 1, 2, 3}
        inp1 = inp.reshape(2, -1)[0]
        inp2 = inp.reshape(2, -1)[1]
        out1 = sample.reshape(2, -1)[0].reshape(-1, 2)
        out2 = sample.reshape(2, -1)[1].reshape(-1, 2)
        return check_answer(inp1, inp2, out1, out2)
    
def check_input_output(inp, output, flatten=True):
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

data = {'input': [], 'output': []}

for i in tqdm(range(n_inp)):
    inp = np.random.randint(4, size=2 * n)
    circ = compute(inp)
    task = device.run(circ, s3_folder, shots=n_shot)
    # task = device.run(circ, shots=n_shot)

    input = np.tile(num_to_bit(inp).reshape(1, -1), (n_shot, 1))
    output = task.result().measurements
    data['input'].append(input)
    data['output'].append(output)

    with open(f'aws-{i}.json', 'w') as f:
        f.write(orjson.dumps({'input': input, 'output': output}, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))

data['input'] = np.vstack(data['input'])
data['output'] = np.vstack(data['output'])

# save via orjson
with open('aws.json', 'w') as f:
    f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8'))

check = check_input_output(data['input'], data['output'], flatten=False)
score = np.mean(np.sum(check, axis=-1) > 0.95 * n)
std = 1.96 * np.sqrt(score * (1 - score) / len(check))
print(score, std, len(data['input']))
print(t.simulator_tasks_cost())