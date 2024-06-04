import numpy as np
import matplotlib.pyplot as plt
import pyclifford as pc
from quantum_model import Quantum_Strategy
from tqdm import tqdm
import orjson
from utils import plot_mean_std_shaded
import scienceplots
plt.style.use(['science','no-latex'])

# n = 100, n_shots = 1000, time = 1h
n_list = [200] # [10, 20, 50, 100]
n_shots = 10000

res = {}
for n in tqdm(n_list):
    key = str(n)
    res[key] = []
    for noise in np.linspace(0, 0.03, 200):
        qs = Quantum_Strategy(n, noise=noise)
        data = qs.produce_data(n_shots, progress_bar=False)
        check = qs.check_input_output(data['X'], data['Y'], flatten=False)
        score = np.mean(np.sum(check, axis=-1) > 0.95 * n)
        std = 1.96 * np.sqrt(score * (1 - score) / len(check)) # 95% confidence interval
        # print(f"noise: {noise:.2f}, score: {score:.2f}")
        res[key].append([noise, score, std])
    res[key] = np.array(res[key])
    
    with open("noisy_quantum_model_200.json", "wb") as f:
        f.write(orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY))

plt.figure(figsize=(6, 4))
for i, n in enumerate(n_list):
    # plt.errorbar(res[str(n)][:, 0], res[str(n)][:, 1], yerr=res[str(n)][:, 2]/10, label=f"n={4*n}", capsize=2)
    plot_mean_std_shaded(res[str(n)][:, 0], res[str(n)][:, 1], res[str(n)][:, 2], color=f'C{i}', label=f"$n={4*n}$")
    # plt.plot(res[n][:, 0], res[n][:, 1], label=f"n={4 * n}")
plt.vlines(1-(15/16)**(0.1), -0.05, 1.05, color='k', linestyle='--', label=r"$p^\star$")
plt.legend()
plt.xlabel(r"Noise")
plt.ylabel(r"Score")
# plt.xscale('log')
plt.ylim(-0.05, 1.05)
# plt.yscale('log')
plt.savefig("noisy_quantum_model.pdf")