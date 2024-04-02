import numpy as np
from quantum_model import Quantum_Strategy
import line_profiler

profiler = line_profiler.LineProfiler()

@profiler
def test(n):
    qs = Quantum_Strategy(n)
    inp = np.random.randint(0, 4, 2*n)
    sample = qs.compute_and_sample(inp)
    answer = qs.produce_answer(inp, sample)
    print(f'acc: {np.mean(answer)}')
    
if __name__ == '__main__':
    test(100)
    profiler.print_stats()