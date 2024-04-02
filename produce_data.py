import argparse
from quantum_model import Quantum_Strategy

def produce_data(n, n_samples, save_path):
    save_path = save_path if save_path is not None else f'./data_{n}.npz'
    qs = Quantum_Strategy(n)
    qs.produce_data(n_samples, save_path=save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce and save data')
    parser.add_argument('--n', type=int, default=100, help='Number of quantum bits')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the generated data')

    args = parser.parse_args()

    produce_data(args.n, args.n_samples, args.save_path)
    
    