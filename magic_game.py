import numpy as np

def check_answer(in_1, in_2, out_1, out_2):
    '''
        in: (n), each {0, 1, 2, 3}
        out: (n, 2), each {0, 1}^2
        check: (n), each {0, 1}
    '''
    # automatic determine the third output bit
    out_1_full = np.zeros((len(in_1), 3), dtype=int)
    out_2_full = np.zeros((len(in_1), 3), dtype=int)
    out_1_full[:, :-1] = out_1
    out_2_full[:, :-1] = out_2
    out_1_full[:, -1] = 1 - (np.sum(out_1, axis=-1) % 2)
    out_2_full[:, -1] = np.sum(out_2, axis=-1) % 2
    # check the thrid output bit
    assert np.sum(np.sum(out_1_full, axis=-1) % 2 == 1) == len(in_1)
    assert np.sum(np.sum(out_2_full, axis=-1) % 2 == 0) == len(in_1)
    
    check = (out_1_full[np.arange(len(in_1)), in_2-1] == out_2_full[np.arange(len(in_1)), in_1-1]).reshape(-1).astype(int)
    
    # if in=0, then succeed directly
    check[in_1 == 0] = 1
    check[in_2 == 0] = 1
    return check

def num_to_bit(num, flatten=True):
    '''
        num: (n), each {0, 1, 2, 3}
        bit: (n, 2), each {0, 1}^2 if not flatten, (2n) if flatten
    '''
    if flatten:
        return (((num[:, None] & (1 << np.arange(2))[::-1])) > 0).astype(int).flatten()
    return (((num[:, None] & (1 << np.arange(2))[::-1])) > 0).astype(int)

def bit_to_num(bit):
    '''
        bit: (n, 2), each {0, 1}^2
        num: (n), each {0, 1, 2, 3}
    '''
    if len(bit.shape) == 1:
        bit = bit.reshape(-1, 2)
    return np.sum(bit * (2 ** np.arange(bit.shape[-1])[::-1]).reshape(1, -1), axis=-1)